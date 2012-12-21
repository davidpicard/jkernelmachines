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
package fr.lip6.jkernelmachines.test.evaluation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import fr.lip6.jkernelmachines.classifier.Classifier;
import fr.lip6.jkernelmachines.classifier.LaSVM;
import fr.lip6.jkernelmachines.evaluation.AccuracyEvaluator;
import fr.lip6.jkernelmachines.evaluation.Evaluator;
import fr.lip6.jkernelmachines.evaluation.LeaveOneOutCrossValidation;
import fr.lip6.jkernelmachines.evaluation.NFoldCrossValidation;
import fr.lip6.jkernelmachines.evaluation.RandomSplitCrossValidation;
import fr.lip6.jkernelmachines.kernel.typed.DoubleGaussL2;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.DebugPrinter;

/**
 * Test cases for the cross-validation related classes.
 * @author picard
 *
 */
public class TestCrossValidation {
	
	static DebugPrinter debug = new DebugPrinter();

	/**
	 * @param args ignored
	 */
	public static void main(String[] args) {

		int dimension = 3;
		int nbPosTrain = 50;
		int nbNegTrain = 50;
		double p = 1.5;

		Random ran = new Random(System.currentTimeMillis());

		ArrayList<TrainingSample<double[]>> train = new ArrayList<TrainingSample<double[]>>();
		// 1. generate positive train samples
		for (int i = 0; i < nbPosTrain; i++) {
			double[] t = new double[dimension];
			for (int x = 0; x < dimension; x++) {
				t[x] = -p + ran.nextDouble();
			}

			train.add(new TrainingSample<double[]>(t, 1));
		}
		// 2. generate negative train samples
		for (int i = 0; i < nbNegTrain; i++) {
			double[] t = new double[dimension];
			for (int x = 0; x < dimension; x++) {
				t[x] = p + ran.nextDouble();
			}

			train.add(new TrainingSample<double[]>(t, -1));
		}
		
		// 2.1 shuffle list to avoid weird behavior
		Collections.shuffle(train, ran);

		// 3. instanciate classifier
		DoubleGaussL2 k = new DoubleGaussL2();
		k.setGamma(1.0);
		LaSVM<double[]> svm = new LaSVM<double[]>(k);
		svm.setC(100);
		
		DebugPrinter.setDebugLevel(0);
		
		int good = 0;
		if(testRandomSplitCrossValidation(svm, train))
			good++;
		else
			System.err.println("WARNING: RandomSplitCrossValidation failed!");
		if(testLeaveOneOutCrossValidation(svm, train))
			good++;
		else
			System.err.println("WARNING: LeaveOneOutCrossValidation failed!");
		if(testNFoldCrossValidation(svm, train))
			good++;
		else
			System.err.println("WARNING: NFoldCrossValidation failed!");
		
		System.out.println("Testing Crossvalidation: "+good+"/3 tests validated.");

	}

	/*
	 * test RandomSplitCrossValidation class
	 */
	private static boolean testRandomSplitCrossValidation(
			Classifier<double[]> c, List<TrainingSample<double[]>> l) {
		
		// 4. CrossValidation
		Evaluator<double[]> eval = new AccuracyEvaluator<double[]>();
		RandomSplitCrossValidation<double[]> cv = new RandomSplitCrossValidation<double[]>(
				c, l, eval);
		cv.setTrainPercent(0.80);
		cv.setNbTest(10);

		// 5. perfom tests
		cv.run();

		// 6. get results
		debug.println(1,"Accuracy: " + cv.getAverageScore() + " +/- "
				+ cv.getStdDevScore());
		debug.println(1,"(scores: " + Arrays.toString(cv.getScores()) + ")");
		
		return (cv.getAverageScore()==1.0);
	}

	/*
	 * test LeaveOneOutCrossValidation class
	 */
	private static boolean testLeaveOneOutCrossValidation(
			Classifier<double[]> c, List<TrainingSample<double[]>> l) {
		
		// 4. CrossValidation
		Evaluator<double[]> eval = new AccuracyEvaluator<double[]>();
		LeaveOneOutCrossValidation<double[]> cv = new LeaveOneOutCrossValidation<double[]>(
				c, l, eval);

		// 5. perfom tests
		cv.run();

		// 6. get results
		debug.println(1,"Accuracy: " + cv.getAverageScore() + " +/- "
				+ cv.getStdDevScore());
		debug.println(1,"(scores: " + Arrays.toString(cv.getScores()) + ")");
		
		return (cv.getAverageScore()==1.0);
	}
	
	/*
	 * test NFoldCrossValidation class
	 */
	private static boolean testNFoldCrossValidation(
			Classifier<double[]> c, List<TrainingSample<double[]>> l) {
		
		// 4. CrossValidation
		Evaluator<double[]> eval = new AccuracyEvaluator<double[]>();
		NFoldCrossValidation<double[]> cv = new NFoldCrossValidation<double[]>(
				5, c, l, eval);

		// 5. perfom tests
		cv.run();

		// 6. get results
		debug.println(1,"Accuracy: " + cv.getAverageScore() + " +/- "
				+ cv.getStdDevScore());
		debug.println(1,"(scores: " + Arrays.toString(cv.getScores()) + ")");
		
		return (cv.getAverageScore()==1.0);
	}
}
