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
package fr.lip6.jkernelmachines.evaluation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import fr.lip6.jkernelmachines.classifier.Classifier;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * <p>Class for simple random split based cross validation. The list of samples is split
 * random into training and validation sets (using the trainPercent parameter). nbTest
 * evaluation are performed.</p>
 * 
 * <p>By default, 70% of the samples are used for training, and 20 tests are performed.</p>
 * This CV is balanced by default.
 * 
 * @author picard
 *
 * @param <T> samples datatype
 */
public class RandomSplitCrossValidation<T> implements CrossValidation, BalancedCrossValidation {

	boolean balance = true;
	
	Classifier<T> classifier;
	List<TrainingSample<T>> list;
	Evaluator<T> evaluator;
	long seed = 0;
	
	
	double trainPercent = 0.7;
	int nbTest = 20;
	
	//results
	double[] results;
	
	/**
	 * Default constructor which should provide a classifier to be tested, 
	 * the complete list of samples and the evaluator computing the scores
	 * @param cls the classifier to be trained and tested
	 * @param l the list of available samples
	 * @param e the evaluator used for the score
	 */
	public RandomSplitCrossValidation(Classifier<T> cls, List<TrainingSample<T>> l, Evaluator<T> e) {
		classifier = cls;
		list = new ArrayList<TrainingSample<T>>();
		list.addAll(l);
		evaluator = e;
	}
	
	@Override
	public void run() {
		int nb = nbTest;
		results = new double[nbTest];
		
		int trainSize = (int) (trainPercent * list.size());
		Random ran = new Random(seed);
		while(nb > 0){
			
			//random split
			Collections.shuffle(list, ran);
			List<TrainingSample<T>> trainList = new ArrayList<>(trainSize);
			List<TrainingSample<T>> testList = new ArrayList<>(list.size()-trainSize);
			if(balance) {
				List<TrainingSample<T>> pos = new ArrayList<TrainingSample<T>>();
				List<TrainingSample<T>> neg = new ArrayList<TrainingSample<T>>();
				for(TrainingSample<T> t : list) {
					if(t.label == 1) {
						pos.add(t);
					}
					else {
						neg.add(t);
					}
				}
				
				trainList.addAll(pos.subList(0, (int)(pos.size()*trainPercent)));
				testList.addAll(pos);
				trainList.addAll(neg.subList(0, (int)(neg.size()*trainPercent)));
				testList.addAll(neg);
				testList.removeAll(trainList);
				
			}
			else {
				trainList.addAll(list.subList(0, trainSize));
				testList.addAll(list);
				testList.removeAll(trainList);
			}
			
			//set evaluator parameters
			evaluator.setClassifier(classifier);
			evaluator.setTrainingSet(trainList);
			evaluator.setTestingSet(testList);
			
			//evaluate
			evaluator.evaluate();
			
			//get score
			results[nbTest-nb] = evaluator.getScore();
			
			nb--;
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

	/**
	 * Tells the classifier
	 * @return the classifier used in the tests
	 */
	public Classifier<T> getClassifier() {
		return classifier;
	}

	/**
	 * Sets the classifier
	 * @param classifier the classifier used in the tests
	 */
	public void setClassifier(Classifier<T> classifier) {
		this.classifier = classifier;
	}

	/**
	 * Tells the list of samples
	 * @return the list of samples used in the test
	 */
	public List<TrainingSample<T>> getList() {
		return list;
	}

	/**
	 * Sets the list of samples
	 * @param list the list of samples to be used in the tests
	 */
	public void setList(List<TrainingSample<T>> list) {
		this.list = list;
	}

	/**
	 * Tells the percentage of samples used for training
	 * @return the percent of available samples to keep for training
	 */
	public double getTrainPercent() {
		return trainPercent;
	}

	/**
	 * Sets the percentage of samples used for training
	 * @param trainPercent the percent of available samples to keep for training
	 */
	public void setTrainPercent(double trainPercent) {
		this.trainPercent = trainPercent;
	}

	/**
	 * Tells the number of tests performed
	 * @return the number of tests performed
	 */
	public int getNbTest() {
		return nbTest;
	}

	/**
	 * Sets the number of tests to perfom
	 * @param nbTest the number of tests to perform
	 */
	public void setNbTest(int nbTest) {
		this.nbTest = nbTest;
	}

	public long getSeed() {
		return seed;
	}

	public void setSeed(long seed) {
		this.seed = seed;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.evaluation.BalancedCrossValidation#isBalanced()
	 */
	@Override
	public boolean isBalanced() {
		return balance;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.evaluation.BalancedCrossValidation#setBalanced(boolean)
	 */
	@Override
	public void setBalanced(boolean balanced) {
		balance = balanced;
	}

}
