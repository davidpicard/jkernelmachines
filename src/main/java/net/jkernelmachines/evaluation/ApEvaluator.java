/*******************************************************************************
 * Copyright (c) 2016, David Picard.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************/
package net.jkernelmachines.evaluation;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import net.jkernelmachines.classifier.Classifier;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.DebugPrinter;

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
	
	DebugPrinter debug = new DebugPrinter();
	
	/**
	 * default constructor
	 */
	public ApEvaluator() {
		
	}
	
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
                if(train != null) {
                    train();
                    debug.println(2, "training done in "+(System.currentTimeMillis()-time)+" ms");
                }
		time = System.currentTimeMillis();
		esResults = evaluateSet(test);
		debug.println(2, "testingset done in "+(System.currentTimeMillis()-time));
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
			debug.println(1, "Evaluator error - result corrupted");
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
