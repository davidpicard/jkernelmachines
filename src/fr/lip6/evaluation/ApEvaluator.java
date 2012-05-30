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
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import fr.lip6.classifier.Classifier;
import fr.lip6.type.TrainingSample;

/**
 * <p>
 * Simple evaluation class for computing the mean average precision, VOC style.
 * </p>
 * <p>
 * Does training, evaluation and timing statistics.
 * </p>
 * @author picard
 *
 * @param <T> datatype of input space
 */
public class ApEvaluator<T> implements Serializable, Evaluator<T>
{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -2713343666983051855L;
	
	Classifier<T> classifier;
	List<TrainingSample<T>> train;
	List<TrainingSample<T>> test;
	List<Evaluation<TrainingSample<T>>> esResults;
	
	/**
	 * Constructor using a classifier, train and test lists.
	 * @param c the classifier
	 * @param trainList the list of training samples
	 * @param testList the list on which to perform the evaluation
	 */
	public ApEvaluator(Classifier<T> c, List<TrainingSample<T>> trainList, List<TrainingSample<T>> testList)
	{
		classifier = c;
		train = trainList;
		test = testList;
		
		//instanciate Evaluation for loading class
		Evaluation<T> e = new Evaluation<T>(null, 0);
		e.compareTo(null);
	}
	
	@Override
	public void evaluate()
	{
		long time = System.currentTimeMillis();
		train();
		System.out.println("training done in "+(System.currentTimeMillis()-time)+" ms");
		time = System.currentTimeMillis();
		esResults = evaluateSet(test);
		System.out.println("testingset done in "+(System.currentTimeMillis()-time));
	}
	
	
	private void train()
	{
		classifier.train(train);
	}
	
	
	/**
	 * Evaluate the classifier on all elements of a set
	 * @param l the list of sample to classify
	 * @return a list containing evaluation of the samples
	 */
	private List<Evaluation<TrainingSample<T>>> evaluateSet(final List<TrainingSample<T>> l) {
		
		final List<Evaluation<TrainingSample<T>>> results = new ArrayList<Evaluation<TrainingSample<T>>>();
	
		//max cpu
		int nbcpu = Runtime.getRuntime().availableProcessors();

		//one job per line of the matrix
		int length = l.size();
		ThreadPoolExecutor threadPool = new ThreadPoolExecutor(nbcpu, nbcpu, 10, TimeUnit.SECONDS, new ArrayBlockingQueue<Runnable>(length+2));
		for(int i = length-1 ; i >= 0 ; i--)
		{
			final int index = i;
			Runnable r = new Runnable(){
				@Override
				public void run() {
					TrainingSample<T> s = l.get(index);
					double r = classifier.valueOf(s.sample);
					Evaluation<TrainingSample<T>> e = new Evaluation<TrainingSample<T>>(s, r);
					synchronized(results)
					{
						results.add(e);
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
	
		return results;
	}
	
	
	// compute map
	private double getMAP(List<Evaluation<TrainingSample<T>>> l)
	{
		if(l == null)
			return Double.NaN;
		
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
		
		//precision / recall
		double[] prec = new double[tp.length];
		double[] reca = new double[tp.length];
		
		for(i = 0 ; i < tp.length ; i++)
		{
			reca[i] = ((double)tp[i])/((double)totalpos);
			prec[i] = ((double)tp[i])/((double)(tp[i]+fp[i]));
		}
		
		//compute map only on 11 points
		double map = 0.;
		i = 0;
		for(double t = 0 ; t <= 1 ; t = t + 0.1)
		{
			while(reca[i] < t)
				i++;
			double pmax = 0;
			for(int j = i ; j < prec.length ; j++)
				if(prec[j] > pmax)
				{
					pmax = prec[j];
				}
			map += pmax/11.;
		}
		
		return map;
	}
	
	@Override
	public void setClassifier(Classifier<T> cls) {
		classifier = cls;
	}

	@Override
	public void setTrainingSet(List<TrainingSample<T>> trainlist) {
		train = trainlist;
	}

	@Override
	public void setTestingSet(List<TrainingSample<T>> testlist) {
		test = testlist;
	}

	@Override
	public double getScore() {
		return getMAP(esResults);
	}

}
