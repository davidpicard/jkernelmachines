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
package fr.lip6.test.evaluation;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import fr.lip6.classifier.Classifier;
import fr.lip6.classifier.LaSVM;
import fr.lip6.evaluation.AccuracyEvaluator;
import fr.lip6.evaluation.ApEvaluator;
import fr.lip6.evaluation.Evaluator;
import fr.lip6.kernel.typed.DoubleGaussL2;
import fr.lip6.type.TrainingSample;
import fr.lip6.util.DebugPrinter;

/**
 * Test cases for the evaluation related classes
 * @author picard
 *
 */
public class TestEvaluation {

	
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

		// 3. instanciate classifier
		DoubleGaussL2 k = new DoubleGaussL2();
		k.setGamma(1.0);
		LaSVM<double[]> svm = new LaSVM<double[]>(k);
		svm.setC(100);
		
		DebugPrinter.setDebugLevel(0);
		
		int good = 0;
		if(testAccuracyEvaluator(svm, train))
			good++;
		else
			System.err.println("WARNING: AccuracyEvaluator failed!");
		
		if(testAveragePrecisionEvaluator(svm, train))
			good++;
		else
			System.err.println("WARNING: ApEvaluator failed!");
		
		System.out.println("Testing Evaluation: "+good+"/2 tests validated.");

	}

	/*
	 * test AccuracyEvaluator class
	 */
	private static boolean testAccuracyEvaluator(
			Classifier<double[]> c, List<TrainingSample<double[]>> l) {
		
		// 4. CrossValidation
		Evaluator<double[]> eval = new AccuracyEvaluator<double[]>();

		// 5. perfom tests
		eval.setClassifier(c);
		eval.setTrainingSet(l);
		eval.setTestingSet(l);
		eval.evaluate();

		// 6. get results
		debug.println(1,"Accuracy: " + eval.getScore());
		
		return (eval.getScore()==1.0);
	}

	/*
	 * test ApEvaluator class
	 */
	private static boolean testAveragePrecisionEvaluator(
			Classifier<double[]> c, List<TrainingSample<double[]>> l) {
		
		// 4. CrossValidation
		Evaluator<double[]> eval = new ApEvaluator<double[]>();

		// 5. perfom tests
		eval.setClassifier(c);
		eval.setTrainingSet(l);
		eval.setTestingSet(l);
		eval.evaluate();

		// 6. get results
		debug.println(1,"Average precision: " + eval.getScore());
		
		return (eval.getScore()>=1.0);
	}
}
