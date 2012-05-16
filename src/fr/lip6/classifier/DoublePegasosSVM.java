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
package fr.lip6.classifier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import fr.lip6.kernel.typed.DoubleLinear;
import fr.lip6.type.TrainingSample;

/**
 * <p>
 * Linear SVM classifier on arrays of double using the PEGASOS algorithm.<br />
 * </p>
 * <p>
 * <b>Pegasos: Primal Estimated sub-GrAdient SOlver for SVM</b><br/>
 * Shai S. Shwartz, Yoram Singer, Nathan Srebro<br/>
 * <i>In Proceedings of the 24th international conference on Machine learning (2007), pp. 807-814.</i>
 * </p>
 * @author picard
 *
 */
public class DoublePegasosSVM implements Classifier<double[]>, Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 5289136605543751554L;
	
	private DoubleLinear kernel = new DoubleLinear();
	private List<TrainingSample<double[]>> tList;
	private double[] w;
	private double b = 0;
	
	
	int T = 100000;
	int k = 10;
	double lambda = 1e-3;
	double t0 = 1.e2;
	boolean bias = true;
	
	double C = 1;
	boolean hasC = false;
	

	
	
	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#train(java.util.ArrayList)
	 */
	@Override
	public void train(List<TrainingSample<double[]>> l) {
				
		//hard limit for k
		if(k > l.size())
			k = l.size();
		
		tList = l;
		int taille = tList.get(0).sample.length; 
		//reset w
		w = new double[taille];
		Arrays.fill(w, 0.);
		
		//reset bias
		b = 0;
		
		//check if C
		if(hasC)
			lambda = 1.0 / (C * tList.size());
		
		eprintln(1, "begin training");
		long time = System.currentTimeMillis();
		List<Integer> intList = new ArrayList<Integer>();
		long tLsize = tList.size();
		for(int in = 0 ; in < tLsize ; in++)
			intList.add(in);
		for(int i = 0; i< T; i++)
		{
			//sub sample selection
			Collections.shuffle(intList);
			ArrayList<Integer> subSampleIndices = new ArrayList<Integer>();
			subSampleIndices.addAll(intList.subList(0, k));
			
			//remove y(<w,x>-b) >= 1
			for(Iterator<Integer> iter = subSampleIndices.iterator(); iter.hasNext(); )
			{
				Integer index = iter.next();
				double[] d = tList.get(index).sample;
				int y = tList.get(index).label;
				if((kernel.valueOf(w, d) - b)*y > 1)
					iter.remove();
			}
			
			//choosing step
			double eta = 1/(double)(lambda*(i+t0));
			
			//calculate half step coefficient;
			double[] w_halfstep = new double[w.length]; 
			double omel = (1-eta*lambda);
			for(int m = 0 ; m < taille; m++)
			{
				w_halfstep[m] = omel*w[m];
			}
			double dir = 0;
			for(Iterator<Integer> iter =  subSampleIndices.iterator(); iter.hasNext(); )
			{
				Integer index = iter.next();
				TrainingSample<double[]> t = tList.get(index);
				for(int m = 0 ; m < taille; m++)
				{
					if(t.sample[m] != 0)
					{
						dir = t.label*t.sample[m];
						w_halfstep[m] += eta/(double)(k)*dir;
					}
				}

			}
			
			//b
			double b_new = 0;
			if(bias)
				for(int index : subSampleIndices)
				{
					b_new += tList.get(index).label;

				}
			
			//final step
			
			double norm = Math.sqrt(kernel.valueOf(w_halfstep, w_halfstep));
			double min = 1/Math.sqrt(lambda)/norm;
			if(min > 1)
				min = 1;
			
			double[] w_fullstep = w_halfstep.clone();
			for(int m = 0 ; m < taille; m++)
				w_fullstep[m] = w_halfstep[m] * min;
			

			
			w = w_fullstep;
			if(bias)
				b = min*( (1-eta*lambda)*b - eta/(double)k*b_new);
			else
				b = 0;

			eprintln(4, "w : "+Arrays.toString(w)+" b : "+b);
			if(T>20 && i%(T/20) == 0)
				eprint(2, ".");
			
		}
		eprintln(2, "");
	
		
		eprintln(1, "done in "+(System.currentTimeMillis()-time)+" ms");
		eprintln(3, "w : "+Arrays.toString(w)+" b : "+b);
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#train(fr.lip6.type.TrainingSample)
	 */
	@Override
	public void train(TrainingSample<double[]> t) {
		if(tList == null)
			tList = new ArrayList<TrainingSample<double[]>>();
		
		tList.add(t);
		
		train(tList);
		
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(double[] e) {
		return kernel.valueOf(w, e)-b;
	}
	



	/**
	 * Tells the maximum number of iteration of this classifier
	 * @return the t
	 */
	public int getT() {
		return T;
	}


	/**
	 * Sets the maximum number of iterations
	 * @param t the maximum number of iteration
	 */
	public void setT(int t) {
		T = t;
	}


	/**
	 * Tells the number of samples used by this classifier to compute the subgradient
	 * @return the k
	 */
	public int getK() {
		return k;
	}


	/**
	 * Sets the number of samples on which to compute the subgradient
	 * @param k the number of samples
	 */
	public void setK(int k) {
		this.k = k;
	}


	/**
	 * Tells the learning rate of this classifier
	 * @return the lambda
	 */
	public double getLambda() {
		return lambda;
	}


	/**
	 * Sets the learning rate lambda
	 * @param lambda the learning rate
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	/**
	 * Tells the hyperplane array of this classifier
	 * @return the hyperplane coordinates
	 */
	public double[] getW() {
		return w;
	}

	/**
	 * Sets the hyperplane coordinates
	 * @param w the w to set
	 */
	public void setW(double[] w) {
		this.w = w;
	}

	/**
	 * Tells the bias b of (w*x -b)
	 * @return the bias
	 */
	public double getB() {
		return b;
	}

	/**
	 * Sets the bias term
	 * @param b the b to set
	 */
	public void setB(double b) {
		this.b = b;
	}



	/**
	 * Tells if this classifier uses a bias term
	 * @return true if a bias is used 
	 */
	public boolean isBias() {
		return bias;
	}

	/**
	 * Sets if the classifier has a bias term
	 * @param bias true for bias, false for non bias
	 */
	public void setBias(boolean bias) {
		this.bias = bias;
	}



	/**
	 * level of verbosity of this classifier
	 */
	private int VERBOSITY_LEVEL = 0;
	
	/**
	 * set how verbose this classifier shall be. <br />
	 * Everything is printed to stderr. <br />
	 * none : 0 (default), few  : 1, more : 2, all : 3
	 * @param l
	 */
	public void setVerbosityLevel(int l)
	{
		VERBOSITY_LEVEL = l;
	}
	
	/**
	 * print errors on error stream if VERBOSITY_LEVEL is greater than level
	 * @param level VERBOSITY_LEVEL required for the display
	 * @param s String to print
	 */
	private void eprint(int level, String s)
	{
		if(VERBOSITY_LEVEL >= level)
			System.err.print(s);
	}
	
	/**
	 * Same as {@eprint} with a line feed.
	 * @param level
	 * @param s
	 */
	private void eprintln(int level, String s)
	{
		if(VERBOSITY_LEVEL >= level)
			System.err.println(s);
	}

	/**
	 * Tells the iteration offset
	 * @return the iteration offset
	 */
	public double getT0() {
		return t0;
	}

	/**
	 * Sets the iteration offset
	 * @param t0 the iteration offset
	 */
	public void setT0(double t0) {
		this.t0 = t0;
	}

	/**
	 * Sets C hyperparameter (automatically converted in lambda)
	 * @param c
	 */
	public void setC(double c)
	{
		hasC = true;
		C = c;
	}
}
