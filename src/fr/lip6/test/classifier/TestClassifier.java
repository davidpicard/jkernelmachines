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
package fr.lip6.test.classifier;

import java.util.ArrayList;
import java.util.Random;

import fr.lip6.classifier.DoublePegasosSVM;
import fr.lip6.classifier.DoubleQNPKL;
import fr.lip6.classifier.DoubleSGD;
import fr.lip6.classifier.DoubleSGDQN;
import fr.lip6.classifier.GradMKL;
import fr.lip6.classifier.LaSVM;
import fr.lip6.classifier.LaSVMI;
import fr.lip6.classifier.SMOSVM;
import fr.lip6.classifier.SimpleMKL;
import fr.lip6.kernel.typed.DoubleGaussL2;
import fr.lip6.kernel.typed.index.IndexDoubleGaussL2;
import fr.lip6.type.TrainingSample;
import fr.lip6.util.DebugPrinter;

/**
 * Test cases for classifier classes
 * @author picard
 * 
 */
public class TestClassifier {
	
	static DebugPrinter debug = new DebugPrinter();
	
	static int dimension = 3;

	/**
	 * @param args ignored
	 */
	public static void main(String[] args) {
		int nbPosTrain = 25;
		int nbNegTrain = 25;
		int nbPosTest = 25;
		int nbNegTest = 25;
		double p = 2.0;

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
		
		ArrayList<TrainingSample<double[]>> test = new ArrayList<TrainingSample<double[]>>();
		//3. generate positive test samples
		for(int i = 0 ; i < nbPosTest; i++)
		{
			double[] t = new double[dimension];
			for(int x = 0 ; x < dimension; x++)
			{
				t[x] = -p + ran.nextDouble();
			}
			
			test.add(new TrainingSample<double[]>(t, 1));
		}
		//4. generate negative test samples
		for(int i = 0 ; i < nbNegTest; i++)
		{
			double[] t = new double[dimension];
			for(int x = 0 ; x < dimension; x++)
			{
				t[x] = p + ran.nextDouble();
			}
			
			test.add(new TrainingSample<double[]>(t, -1));
		}

		DebugPrinter.setDebugLevel(0);
		
		int good = 0;
		//5. test SMO
		if(evaluateSMO(train, test)) {
			good++;
		}
		else {
			debug.println(0, "Warning SMO failed");
		}
		//6. test LaSVM
		if(evaluateLaSVM(train, test)) {
			good++;
		}
		else {
			debug.println(0, "Warning LaSVM failed");
		}
		//7. test LaSVM-I
		if(evaluateLaSVMI(train, test)) {
			good++;
		}
		else {
			debug.println(0, "Warning LaSVM failed");
		}
		//8. test SimpleMKL
		if(evaluateSimpleMKL(train, test)) {
			good++;
		}
		else {
			debug.println(0, "Warning SimpleMKL failed");
		}
		//9. test GradMKL
		if(evaluateGradMKL(train, test)) {
			good++;
		}
		else {
			debug.println(0, "Warning GradMKL failed");
		}
		//10. test QNPKL
		if(evaluateQNPKL(train, test)) {
			good++;
		}
		else {
			debug.println(0, "Warning QNPKL failed");
		}
		//11. test SGD
		if(evaluateSGD(train, test)) {
			good++;
		}
		else {
			debug.println(0, "Warning SGD failed");
		}
		//12. test SGDQN
		if(evaluateSGDQN(train, test)) {
			good++;
		}
		else {
			debug.println(0, "Warning SGDQN failed");
		}
		//13. test Pegasos
		if(evaluatePegasos(train, test)) {
			good++;
		}
		else {
			debug.println(0, "Warning Pegasos failed");
		}
		
		debug.println(0, "Testing classifiers: "+good+"/9 tests validated");

	}

	private static boolean evaluateSMO(
			ArrayList<TrainingSample<double[]>> train,
			ArrayList<TrainingSample<double[]>> test) {
		
		DoubleGaussL2 k = new DoubleGaussL2();
		k.setGamma(0.05);
		SMOSVM<double[]> svm = new SMOSVM<double[]>(k);
		svm.setC(10);
		svm.train(train);
		
		for(TrainingSample<double[]> t : test)
			if(svm.valueOf(t.sample) * t.label <= 0) {
				debug.println(0, "error with sample "+t+" expected "+t.label+", got "+svm.valueOf(t.sample));
				return false;
			}
		
		return true;
	}

	private static boolean evaluateLaSVM(
			ArrayList<TrainingSample<double[]>> train,
			ArrayList<TrainingSample<double[]>> test) {
		
		DoubleGaussL2 k = new DoubleGaussL2();
		k.setGamma(0.05);
		LaSVM<double[]> svm = new LaSVM<double[]>(k);
		svm.setC(10);
		svm.train(train);
		
		for(TrainingSample<double[]> t : test)
			if(svm.valueOf(t.sample) * t.label <= 0) {
				debug.println(0, "error with sample "+t+" expected "+t.label+", got "+svm.valueOf(t.sample));
				return false;
			}
		
		return true;
	}

	private static boolean evaluateLaSVMI(
			ArrayList<TrainingSample<double[]>> train,
			ArrayList<TrainingSample<double[]>> test) {
		
		DoubleGaussL2 k = new DoubleGaussL2();
		k.setGamma(0.05);
		LaSVMI<double[]> svm = new LaSVMI<double[]>(k);
		svm.setC(10);
		svm.train(train);
		
		for(TrainingSample<double[]> t : test)
			if(svm.valueOf(t.sample) * t.label <= 0) {
				debug.println(0, "error with sample "+t+" expected "+t.label+", got "+svm.valueOf(t.sample));
				return false;
			}
		
		return true;
	}

	private static boolean evaluateSimpleMKL(
			ArrayList<TrainingSample<double[]>> train,
			ArrayList<TrainingSample<double[]>> test) {
		
		//test default setting
		SimpleMKL<double[]> svm = new SimpleMKL<double[]>();
		for(int i = 0 ; i < dimension ; i++){
			svm.addKernel(new IndexDoubleGaussL2(i));
		}
		svm.setC(10);
		svm.train(train);
		
		for(TrainingSample<double[]> t : test)
			if(svm.valueOf(t.sample) * t.label <= 0) {
				debug.println(0, "SimpleMKL(LASVM) error with sample "+t+" expected "+t.label+", got "+svm.valueOf(t.sample));
				return false;
			}
		
		// test with SMO
		svm = new SimpleMKL<double[]>();
		svm.setClassifier(new SMOSVM<double[]>(null));
		for(int i = 0 ; i < dimension ; i++){
			svm.addKernel(new IndexDoubleGaussL2(i));
		}
		svm.setC(10);
		svm.train(train);
		
		for(TrainingSample<double[]> t : test)
			if(svm.valueOf(t.sample) * t.label <= 0) {
				debug.println(0, "SimpleMKL(SMOSVM) error with sample "+t+" expected "+t.label+", got "+svm.valueOf(t.sample));
				return false;
			}
		
		// test with LaSVM-I
		svm = new SimpleMKL<double[]>();
		svm.setClassifier(new LaSVMI<double[]>(null));
		for(int i = 0 ; i < dimension ; i++){
			svm.addKernel(new IndexDoubleGaussL2(i));
		}
		svm.setC(10);
		svm.train(train);
		
		for(TrainingSample<double[]> t : test)
			if(svm.valueOf(t.sample) * t.label <= 0) {
				debug.println(0, "SimpleMKL(LASVMI) error with sample "+t+" expected "+t.label+", got "+svm.valueOf(t.sample));
				return false;
			}
		
		return true;
	}

	private static boolean evaluateGradMKL(
			ArrayList<TrainingSample<double[]>> train,
			ArrayList<TrainingSample<double[]>> test) {
		
		// default SMO
		GradMKL<double[]> svm = new GradMKL<double[]>();
		for(int i = 0 ; i < dimension ; i++){
			svm.addKernel(new IndexDoubleGaussL2(i));
		}
		svm.setC(10);
		svm.train(train);
		
		for(TrainingSample<double[]> t : test)
			if(svm.valueOf(t.sample) * t.label <= 0) {
				debug.println(0, "GradMKL(SMOSVM) error with sample "+t+" expected "+t.label+", got "+svm.valueOf(t.sample));
				return false;
			}
		
		//LaSVM
		svm = new GradMKL<double[]>();
		svm.setClassifier(new LaSVM<double[]>(null));
		for(int i = 0 ; i < dimension ; i++){
			svm.addKernel(new IndexDoubleGaussL2(i));
		}
		svm.setC(10);
		svm.train(train);
		
		for(TrainingSample<double[]> t : test)
			if(svm.valueOf(t.sample) * t.label <= 0) {
				debug.println(0, "GradMKL(LASVM) error with sample "+t+" expected "+t.label+", got "+svm.valueOf(t.sample));
				return false;
			}

		//LaSVMI
		svm = new GradMKL<double[]>();
		svm.setClassifier(new LaSVMI<double[]>(null));
		for(int i = 0 ; i < dimension ; i++){
			svm.addKernel(new IndexDoubleGaussL2(i));
		}
		svm.setC(10);
		svm.train(train);
		
		for(TrainingSample<double[]> t : test)
			if(svm.valueOf(t.sample) * t.label <= 0) {
				debug.println(0, "GradMKL(LASVMI) error with sample "+t+" expected "+t.label+", got "+svm.valueOf(t.sample));
				return false;
			}
		
		return true;
	}

	private static boolean evaluateQNPKL(
			ArrayList<TrainingSample<double[]>> train,
			ArrayList<TrainingSample<double[]>> test) {
		
		DoubleQNPKL svm = new DoubleQNPKL();
		svm.setC(10);
		svm.train(train);
		
		for(TrainingSample<double[]> t : test)
			if(svm.valueOf(t.sample) * t.label <= 0) {
				debug.println(0, "error with sample "+t+" expected "+t.label+", got "+svm.valueOf(t.sample));
				return false;
			}
		
		return true;
	}
	
	private static boolean evaluateSGD(
			ArrayList<TrainingSample<double[]>> train,
			ArrayList<TrainingSample<double[]>> test) {
		
		DoubleSGD svm = new DoubleSGD();
		svm.setLambda(1e-2);
		svm.train(train);
		
		for(TrainingSample<double[]> t : test)
			if(svm.valueOf(t.sample) * t.label <= 0) {
				debug.println(0, "error with sample "+t+" expected "+t.label+", got "+svm.valueOf(t.sample));
				return false;
			}
		
		return true;
	}
	
	private static boolean evaluateSGDQN(
			ArrayList<TrainingSample<double[]>> train,
			ArrayList<TrainingSample<double[]>> test) {
		
		DoubleSGDQN svm = new DoubleSGDQN();
		svm.setLambda(1e-2);
		svm.train(train);
		
		for(TrainingSample<double[]> t : test)
			if(svm.valueOf(t.sample) * t.label <= 0) {
				debug.println(0, "error with sample "+t+" expected "+t.label+", got "+svm.valueOf(t.sample));
				return false;
			}
		
		return true;
	}
	
	private static boolean evaluatePegasos(
			ArrayList<TrainingSample<double[]>> train,
			ArrayList<TrainingSample<double[]>> test) {
		
		DoublePegasosSVM svm = new DoublePegasosSVM();
		svm.setLambda(1e-2);
		svm.train(train);
		
		for(TrainingSample<double[]> t : test)
			if(svm.valueOf(t.sample) * t.label <= 0) {
				debug.println(0, "error with sample "+t+" expected "+t.label+", got "+svm.valueOf(t.sample));
				return false;
			}
		
		return true;
	}
}
