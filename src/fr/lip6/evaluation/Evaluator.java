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
package fr.lip6.evaluation;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import fr.lip6.classifier.Classifier;
import fr.lip6.type.TrainingSample;

/**
 * <p>
 * Simple evaluation class for computing the mean average precision.
 * </p>
 * <p>
 * Does training, evaluation and timing statistics.
 * </p>
 * @author picard
 *
 * @param <T> datatype of input space
 */
public class Evaluator<T> implements Serializable 
{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -2713343666983051855L;
	
	Classifier<T> classifier;
	List<TrainingSample<T>> train;
	List<TrainingSample<T>> test;
	List<Evaluation<TrainingSample<T>>> tsResults;
	List<Evaluation<TrainingSample<T>>> esResults;
	
	/**
	 * Constructor using a classifier, train and test lists.
	 * @param c the classifier
	 * @param trainList the list of training samples
	 * @param testList the list on which to perform the evaluation
	 */
	public Evaluator(Classifier<T> c, List<TrainingSample<T>> trainList, List<TrainingSample<T>> testList)
	{
		classifier = c;
		train = trainList;
		test = testList;
		
		//instanciate Evaluation for loading class
		Evaluation<T> e = new Evaluation<T>(null, 0);
		e.compareTo(null);
	}
	
	/**
	 * Train and evaluate results
	 */
	public void evaluate()
	{
		long time = System.currentTimeMillis();
		train();
		System.out.println("training done in "+(System.currentTimeMillis()-time)+" ms");
		time = System.currentTimeMillis();
		evaluateTrainingSet();
		System.out.println("trainingset done in "+(System.currentTimeMillis()-time));
		time = System.currentTimeMillis();
		evaluateTestingSet();
		System.out.println("testingset done in "+(System.currentTimeMillis()-time));
	}
	
	
	private void train()
	{
		classifier.train(train);
	}
	
	/**
	 * Computes output values for each element of the training set
	 */
	private void evaluateTrainingSet()
	{
		if(tsResults == null)
			tsResults = new ArrayList<Evaluation<TrainingSample<T>>>();

		//max cpu
		final int nbcpu = Runtime.getRuntime().availableProcessors();

		//one job per cpu
		ThreadPoolExecutor threadPool = new ThreadPoolExecutor(nbcpu, nbcpu, 10, TimeUnit.MINUTES, new ArrayBlockingQueue<Runnable>(nbcpu));
		final Queue<TrainingSample<T>> q = new LinkedList<TrainingSample<T>>();
		for(int s = 0 ; s < train.size() ; s++)
			q.add(train.get(s));
		for(int i = 0 ; i < nbcpu ; i++)
		{

			final Queue<TrainingSample<T>> mq = new LinkedList<TrainingSample<T>>();
			for(int s = 0 ; s < train.size()/nbcpu+1 ; s++)
				if(!q.isEmpty())
					mq.add(q.remove());
			Runnable r = new Runnable(){

				@Override
				public void run() {

					List<Evaluation<TrainingSample<T>>> mList = new ArrayList<Evaluation<TrainingSample<T>>>();
					while(!mq.isEmpty())
					{
						TrainingSample<T> s = mq.remove();
						double r = classifier.valueOf(s.sample);
						Evaluation<TrainingSample<T>> e = new Evaluation<TrainingSample<T>>(s, r);
						mList.add(e);
					}
					
					synchronized(tsResults)
					{
						tsResults.addAll(mList);
					}
				}
			};
			
			threadPool.execute(r);
		}

		threadPool.shutdown();
		try {
			threadPool.awaitTermination(Integer.MAX_VALUE, TimeUnit.DAYS);
		} catch (InterruptedException e) {
			System.err.println("Evaluator error - result corrupted");
			e.printStackTrace();
		}
		
	}
	
	/**
	 * Computes output values for each of the testing set
	 * @param testingSet
	 */
	private void evaluateTestingSet()
	{
		if(esResults == null)
			esResults = new ArrayList<Evaluation<TrainingSample<T>>>();
	
		//max cpu
		int nbcpu = Runtime.getRuntime().availableProcessors();

		//one job per line of the matrix
		int length = test.size();
		ThreadPoolExecutor threadPool = new ThreadPoolExecutor(nbcpu, nbcpu, 10, TimeUnit.SECONDS, new ArrayBlockingQueue<Runnable>(length+2));
		for(int i = length-1 ; i >= 0 ; i--)
		{
			final int index = i;
			Runnable r = new Runnable(){
				@Override
				public void run() {
					TrainingSample<T> s = test.get(index);
					double r = classifier.valueOf(s.sample);
					Evaluation<TrainingSample<T>> e = new Evaluation<TrainingSample<T>>(s, r);
					synchronized(esResults)
					{
						esResults.add(e);
					}
				}
			};
			
			threadPool.execute(r);
		}

		threadPool.shutdown();
		try {
			threadPool.awaitTermination(Integer.MAX_VALUE, TimeUnit.DAYS);
		} catch (InterruptedException e) {
			System.err.println("Evaluator error - result corrupted");
			e.printStackTrace();
		}
	
		
	}
	
	/**
	 * Computes Mean Average Precision for the training set (VOC style)
	 * @return the MAP
	 */
	public double getTrainingMAP()
	{
		return getMAP(tsResults);
	}
	
	// compute map
	private double getMAP(List<Evaluation<TrainingSample<T>>> l)
	{
		Collections.sort(l);
		
		int[] tp = new int[l.size()];
		int[] fp = new int[l.size()];
		
		int i = 0;
		int cumtp = 0, cumfp = 0;
		int totalpos = 0;
		
		//cumsum of true positives and false positives
		for(Evaluation<TrainingSample<T>> e : l)
		{
			if(e.sample.label == 1)
			{
				cumtp++;
				totalpos++;
			}
			else
			{
				cumfp++;
			}
			tp[i] = cumtp;
			fp[i] = cumfp;
			i++;
		}
		
//		System.out.println("tp and fp done");
//		System.out.println("tp : "+Arrays.toString(tp));
//		System.out.println("totalpos : "+totalpos);
		
		//precision / recall
		double[] prec = new double[tp.length];
		double[] reca = new double[tp.length];
		
		for(i = 0 ; i < tp.length ; i++)
		{
			reca[i] = ((double)tp[i])/((double)totalpos);
			prec[i] = ((double)tp[i])/((double)(tp[i]+fp[i]));
		}
		
//		System.out.println("prec and reca done");
//		System.out.println("prec : "+Arrays.toString(prec));
//		System.out.println("reca : "+Arrays.toString(reca));
		
		//compute map only on 11 points
		double map = 0.;
		i = 0;
		for(double t = 0 ; t <= 1 ; t = t + 0.1)
		{
			while(reca[i] < t)
				i++;
//			System.out.println("t : "+t+" i : "+i);
			double pmax = 0;
			for(int j = i ; j < prec.length ; j++)
				if(prec[j] > pmax)
				{
					pmax = prec[j];
//					System.out.println("i : "+i+" pmax : "+pmax);
				}
			map += pmax/11.;
			
//			System.out.println("on t = "+t+" map = "+map);
		}
		
		return map;
	}
	
	/**
	 * computes the Mean average precision for the testing set (VOC style)
	 * @return the MAP
	 */
	public double getTestingMAP()
	{
		return getMAP(esResults);
	}
	
	/**
	 * Computes the precision curve for the training set
	 * @return
	 */
	public double[] getTrainingPrecision()
	{
		ArrayList<Double> precision = new ArrayList<Double>();
		
		Collections.sort(tsResults);
		int top = 0;
		int i = 1;
		for(Evaluation<TrainingSample<T>> e : tsResults)
		{
			if(e.sample.label == 1)
			{
				top++;
				precision.add(top/(double)i);
			}
			i++;
		}
		
		double[] d = new double[precision.size()];
		for(int j = 0 ; j < precision.size(); j++)
			d[j] = precision.get(j);
		
		return d;
	}
	
	/**
	 * Computes the precision curve for the testing set
	 * @return
	 */
	public double[] getTestingPrecision()
	{
		ArrayList<Double> precision = new ArrayList<Double>();
		
		Collections.sort(esResults);
		int top = 0;
		int i = 1;
		for(Evaluation<TrainingSample<T>> e : esResults)
		{
			if(e.sample.label == 1)
			{
				top++;
				precision.add(top/(double)i);
			}
			i++;
		}
		
		double[] d = new double[precision.size()];
		for(int j = 0 ; j < precision.size(); j++)
			d[j] = precision.get(j);
		
		return d;
	}
	
	/**
	 * returns a map of samples and their associated values for the testing set
	 * @return
	 */
	public HashMap<T, Double> getTestingValues()
	{
		HashMap<T, Double> map = new HashMap<T, Double>();
		for(Evaluation<TrainingSample<T>> e : esResults)
			map.put(e.sample.sample, e.value);
		
		return map;
	}
	
	
	/**
	 * Simple class containing a sample and its evaluation by the classifier
	 * @author dpicard
	 *
	 * @param <U>
	 */
	private class Evaluation<U> implements Comparable<Evaluation<U>>, Serializable
	{
		
		/**
		 * 
		 */
		private static final long serialVersionUID = 791024170617779718L;
		
		U sample;
		double value;
		
		public Evaluation(U s, double v)
		{
			sample = s;
			value = v;
		}
		
		@Override
		public int compareTo(Evaluation<U> o) {
			if(o == null)
				return 0;
			return (int) Math.signum(o.value - value);
		}
	}

}
