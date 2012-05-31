package fr.lip6.test.evaluation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

import fr.lip6.classifier.LaSVM;
import fr.lip6.evaluation.AccuracyEvaluator;
import fr.lip6.evaluation.Evaluator;
import fr.lip6.evaluation.RandomSplitCrossValidation;
import fr.lip6.kernel.typed.DoubleGaussL2;
import fr.lip6.type.TrainingSample;

public class TestCrossValidation {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		int dimension = 3;
		int nbPosTrain = 500;
		int nbNegTrain = 500;
		double p = 2.0;
		
		Random ran = new Random(System.currentTimeMillis());
		
		ArrayList<TrainingSample<double[]>> train = new ArrayList<TrainingSample<double[]>>();
		//1. generate positive train samples
		for(int i = 0 ; i < nbPosTrain; i++)
		{
			double[] t = new double[dimension];
			for(int x = 0 ; x < dimension; x++)
			{
				t[x] = -p + ran.nextGaussian();
			}
			
			train.add(new TrainingSample<double[]>(t, 1));
		}
		//2. generate negative train samples
		for(int i = 0 ; i < nbNegTrain; i++)
		{
			double[] t = new double[dimension];
			for(int x = 0 ; x < dimension; x++)
			{
				t[x] = p + ran.nextGaussian();
			}
			
			train.add(new TrainingSample<double[]>(t, -1));
		}
		
		//3. instanciate classifier
		DoubleGaussL2 k = new DoubleGaussL2();
		k.setGamma(1.0);
		LaSVM<double[]> svm = new LaSVM<double[]>(k);
		svm.setC(100);
		
		//4. CrossValidation
		Evaluator<double[]> eval = new AccuracyEvaluator<double[]>();
		RandomSplitCrossValidation<double[]> cv = new RandomSplitCrossValidation<double[]>(svm, train, eval);
		cv.setTrainPercent(0.75);
		cv.setNbTest(25);
		
		//5. perfom tests
		cv.run();
		
		//6. get results
		System.out.println("Accuracy: "+cv.getAverageScore()+" +/- "+cv.getStdDevScore());
		System.out.println("(scores: "+Arrays.toString(cv.getScores())+")");

	}

}
