/**
    This file is part of JkernelMachines.

    JkernelMachines is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    JkernelMachines is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with JkernelMachines.  If not, see <http://www.gnu.org/licenses/>.

    Copyright David Picard - 2010

*/
package fr.lip6.jkernelmachines.classifier.transductive;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedSet;
import java.util.TreeSet;

import fr.lip6.jkernelmachines.classifier.DoublePegasosSVM;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.DebugPrinter;

/**
 * Fast linear transductive SVM using a combination of SVMLight and Pegasos algorithms.
 * 
 * @author picard
 *
 */
public class S3VMLightPegasos implements TransductiveClassifier<double[]> {

	
	int numplus = 0;
	
	ArrayList<TrainingSample<double[]>> train;
	ArrayList<TrainingSample<double[]>> test;
	
	DoublePegasosSVM svm;
	
	//pegasos parameters
	int T = 100000;
	int k = 10;
	double lambda = 1e-3;
	double t0 = 1.e2;
	boolean bias = true;
	
	DebugPrinter debug = new DebugPrinter();
	
	/**
	 * Default constructor
	 */
	public S3VMLightPegasos()
	{
	}
	
	@Override
	public void train(List<TrainingSample<double[]>> trainList,
			List<TrainingSample<double[]>> testList) {
	
		train = new ArrayList<TrainingSample<double[]>>();
		train.addAll(trainList);
		
		test = new ArrayList<TrainingSample<double[]>>();
		//copy test samples
		for(TrainingSample<double[]> tm : testList)
		{
			TrainingSample<double[]> t = new TrainingSample<double[]>(tm.sample, 0);
			test.add(t);
		}
		
		train();

	}

	private void train()
	{
		debug.println(2, "training on "+train.size()+" train data and "+test.size()+" test data");
		
		//first training
		debug.print(3, "first training ");
		svm = new DoublePegasosSVM();
		svm.setLambda(lambda);
		svm.setK(k);
		svm.setT(T);
		svm.setT0(t0);
		svm.train(train);
		debug.println(3, " done.");
		
		//affect numplus highest output to plus class
		debug.println(3, "affecting 1 to the "+numplus+" highest output");
		SortedSet<TrainingSample<double[]>> sorted = new TreeSet<TrainingSample<double[]>>(new Comparator<TrainingSample<double[]>>(){

			@Override
			public int compare(TrainingSample<double[]> o1, TrainingSample<double[]> o2) {
				int ret = (new Double(svm.valueOf(o2.sample))).compareTo(svm.valueOf(o1.sample));
				if(ret == 0)
					ret = -1;
				return ret;
			}
			
		});
		sorted.addAll(test);
		debug.println(4, "sorted size : "+sorted.size()+" test size : "+test.size());
		int n = 0;
		for(TrainingSample<double[]> t : sorted)
		{
			if(n < numplus)
				t.label = 1;
			else
				t.label = -1;
			n++;
		}
		
		double C = 1. / (train.size()*lambda) ;
		double Cminus = 1e-5;
		double Cplus = 1e-5 * numplus/(test.size() - numplus);
		
		while(Cminus < C || Cplus < C)
		{
			//solve full problem
			ArrayList<TrainingSample<double[]>> full = new ArrayList<TrainingSample<double[]>>();
			full.addAll(train);
			full.addAll(test);
			
			debug.print(3, "full training ");
			svm = new DoublePegasosSVM();
			svm.setLambda(lambda);
			svm.setK(k);
			svm.setT(T);
			svm.setT0(t0);
			svm.train(full);
			debug.println(3, "done.");
			
			boolean changed = false;
			
			do
			{
				changed = false;
				//0. computing error
				final Map<TrainingSample<double[]>, Double> errorCache = new HashMap<TrainingSample<double[]>, Double>();
				for(TrainingSample<double[]> t : test)
				{
					double err1 = 1. - t.label * svm.valueOf(t.sample);
					errorCache.put(t, err1);
				}
				debug.println(3, "Error cache done.");
				
				// 1 . sort by descending error
				sorted = new TreeSet<TrainingSample<double[]>>(new Comparator<TrainingSample<double[]>>(){

					@Override
					public int compare(TrainingSample<double[]> o1,
							TrainingSample<double[]> o2) {
						int ret = errorCache.get(o2).compareTo(errorCache.get(o1));
						if(ret == 0)
							ret = -1;
						return ret;
					}
				});
				sorted.addAll(test);
				List<TrainingSample<double[]>> sortedList = new ArrayList<TrainingSample<double[]>>();
				sortedList.addAll(sorted);
				
				
				debug.println(3, "sorting done, checking couple");
				
				// 2 . test all couple by decreasing error order
//				for(TrainingSample<T> i1 : sorted)
				for(int i = 0 ; i < sortedList.size(); i++)
				{
					TrainingSample<double[]> i1 = sortedList.get(i);
//					for(TrainingSample<T> i2 : sorted)
					for(int j = i+1; j < sortedList.size(); j++)
					{
						TrainingSample<double[]> i2 = sortedList.get(j);
						if(examine(i1, i2, errorCache))
						{
							debug.println(3, "couple found !");
							changed = true;
							break;
						}
					}
					if(changed)
						break;
				}

				if(changed)
				{
					debug.println(3, "re-training");
					svm = new DoublePegasosSVM();
					svm.setLambda(lambda);
					svm.setK(k);
					svm.setT(T);
					svm.setT0(t0);
					svm.train(full);
				}
			}
			while(changed);

			debug.println(3, "increasing C+ : "+Cplus+" and C- : "+Cminus);
			Cminus = Math.min(2*Cminus, C);
			Cplus = Math.min(2 * Cplus, C);
		}
		
		debug.println(2, "training done");
	}
	

	//check if the pair of example fulfill the swapping conditions
	private boolean examine(TrainingSample<double[]> i1, TrainingSample<double[]> i2, Map<TrainingSample<double[]>, Double> errorCache)
	{
		if(i1.label * i2.label > 0)
			return false;
		
		if(!errorCache.containsKey(i1))
			return false;
		double err1 = errorCache.get(i1);	
		if(err1 <= 0)
			return false;
		
		if(!errorCache.containsKey(i2))
			return false;
		double err2 = errorCache.get(i2);
		if(err2 <= 0)
			return false;
		
		debug.println(4, "y1 : "+i1.label+" err1 : "+err1+" y2 : "+i2.label+" err2 : "+err2);
		if(err1 + err2 <= 2)
			return false;
		
		//found a good couple
		int tmplabel = i1.label;
		i1.label = i2.label;
		i2.label = tmplabel;
		
		return true;
	}
	
	
	@Override
	public double valueOf(double[] t) {
		return svm.valueOf(t);
	}

	/**
	 * Tells the number of iteration for internal Pegasos algorithm
	 * @return the number of iterations
	 */
	public int getT() {
		return T;
	}

	/**
	 * Sets the number of iteration for internal Pegasos algorithm
	 * @param t the number of iterations
	 */
	public void setT(int t) {
		T = t;
	}

	/**
	 * Tells the number of samples used for sub-gradient calculation by internal Pegasos solver
	 * @return the number of samples
	 */
	public int getK() {
		return k;
	}

	/**
	 * Sets the number of  samples used for sub-gradient calculation by internal Pegasos solver
	 * @param k the number of samples
	 */
	public void setK(int k) {
		this.k = k;
	}

	/**
	 * Tells the learning rate lambda of internal Pegasos solver
	 * @return the learning rate lambda
	 */
	public double getLambda() {
		return lambda;
	}

	/**
	 * Sets the learning rate lambda of internal Pegasos solver
	 * @param lambda the learning rate
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	/**
	 * Tells the iterations offset of internal Pegasos solver
	 * @return the iteration offset
	 */
	public double getT0() {
		return t0;
	}

	/**
	 * Sets the iterations offset of internal Pegasos solver
	 * @param t0 the iteration offset
	 */
	public void setT0(double t0) {
		this.t0 = t0;
	}

	/**
	 * Tells if this classifier uses a bias term
	 * @return true if using a bias term
	 */
	public boolean isBias() {
		return bias;
	}

	/**
	 * Sets the use of a bias term in this classifier
	 * @param bias true if using a bias
	 */
	public void setBias(boolean bias) {
		this.bias = bias;
	}

	/**
	 * Tells the number of positive samples (used for transductive label estimation)
	 * @return the number of positive samples
	 */
	public int getNumplus() {
		return numplus;
	}

	/**
	 * Sets the number of positives samples (used for transductive label estimation)
	 * @param numplus the number of positive samples
	 */
	public void setNumplus(int numplus) {
		this.numplus = numplus;
	}

	/**
	 * Tells the hyperplane array of this classifier
	 * @return the hyperplane coordinates
	 */
	public double[] getW() {
		return svm.getW();
	}

	/**
	 * Tells the bias b of (w*x -b)
	 * @return the bias
	 */
	public double getB() {
		return svm.getB();
	}

	
}
