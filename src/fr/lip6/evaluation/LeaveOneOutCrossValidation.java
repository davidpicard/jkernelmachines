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

    Copyright David Picard - 2012

*/
package fr.lip6.evaluation;

import java.util.ArrayList;
import java.util.List;

import fr.lip6.classifier.Classifier;
import fr.lip6.type.TrainingSample;

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
