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

import java.util.ArrayList;
import java.util.List;

import net.jkernelmachines.classifier.Classifier;
import net.jkernelmachines.type.TrainingSample;

/**
 * Simple class to perform leave one out cross validation
 * 
 * @author picard
 * 
 * @param <T> samples data type
 *
 */
public class LeaveOneOutCrossValidation<T> implements CrossValidation {

	Classifier<T> classifier;
	List<TrainingSample<T>> list;
	Evaluator<T> evaluator;
		
	//results
	double[] results;
	
	public LeaveOneOutCrossValidation(Classifier<T> cls, List<TrainingSample<T>> l, Evaluator<T> eval) {
		classifier = cls;
		list = new ArrayList<TrainingSample<T>>();
		list.addAll(l);
		evaluator = eval;
	}
	
	
	/* (non-Javadoc)
	 * @see fr.lip6.evaluation.CrossValidation#run()
	 */
	@Override
	public void run() {
		
		results = new double[list.size()];
		
		for(int i = 0 ; i < list.size(); i++) {
			List<TrainingSample<T>> trainList = new ArrayList<TrainingSample<T>>();
			trainList.addAll(list);
			
			TrainingSample<T> t = trainList.remove(i);
			
			List<TrainingSample<T>> testList = new ArrayList<TrainingSample<T>>();
			testList.add(t);
			
			//set evaluator
			evaluator.setClassifier(classifier);
			evaluator.setTrainingSet(trainList);
			evaluator.setTestingSet(testList);
			
			//proceed 
			evaluator.evaluate();
			
			//store score
			results[i] = evaluator.getScore();
		}

	}

	@Override
	public double getAverageScore() {
		
		if(results == null)
			return Double.NaN;
		
		double ave = 0;
		
		for(double d : results)
			ave += d;
		
		return ave/results.length;
	}

	@Override
	public double getStdDevScore() {
		if(results == null)
			return Double.NaN;
		
		double std = 0;
		double ave = getAverageScore();
		
		for(double d : results)
			std += (d-ave)*(d-ave);
		
		return Math.sqrt(std/results.length);
	}

	@Override
	public double[] getScores() {
		return results;
	}

}
