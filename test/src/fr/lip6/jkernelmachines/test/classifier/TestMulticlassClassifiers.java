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
package fr.lip6.jkernelmachines.test.classifier;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import fr.lip6.jkernelmachines.classifier.LaSVM;
import fr.lip6.jkernelmachines.classifier.multiclass.OneAgainstAll;
import fr.lip6.jkernelmachines.kernel.typed.DoubleGaussL2;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.DebugPrinter;

/**
 * @author picard
 * 
 */
public class TestMulticlassClassifiers {

	static DebugPrinter debug = new DebugPrinter();


	/**
	 * @param args
	 */
	public static void main(String[] args) {

		int nbclasses = 5;
		int nbTrainPerClass = 25;
		int nbTestPerClass = 25;
		double p = 4.0;

		Random ran = new Random(System.currentTimeMillis());

		ArrayList<TrainingSample<double[]>> train = new ArrayList<TrainingSample<double[]>>();
		// 1. generate train samples
		for (int c = 1; c <= nbclasses; c++) {
			for (int i = 0; i < nbTrainPerClass; i++) {
				double[] t = new double[nbclasses];
				for (int x = 0; x < nbclasses; x++) {
					t[x] = (x==c)?p:0 + ran.nextDouble();
				}

				train.add(new TrainingSample<double[]>(t, c));
			}
		}

		ArrayList<TrainingSample<double[]>> test = new ArrayList<TrainingSample<double[]>>();
		// 2. generate test samples
		for (int c = 1; c <= nbclasses; c++) {
			for (int i = 0; i < nbTestPerClass; i++) {
				double[] t = new double[nbclasses];
				for (int x = 0; x < nbclasses; x++) {
					t[x] = (x==c)?p:0 + ran.nextDouble();
				}

				test.add(new TrainingSample<double[]>(t, c));
			}
		}
		
		//3. perform tests
		int nbgood = 0;
		if(testOneAgainstAll(train, test)) {
			nbgood++;
		}
		else {
			debug.println(0, "Warning OneAgainstAll failed!");
		}
		
		//show summary
		debug.println(0, "Testing multiclass classifiers: "+nbgood+"/1 test validated.");
	}

	private static boolean testOneAgainstAll(List<TrainingSample<double[]>> train, List<TrainingSample<double[]>> test) {
		
		DoubleGaussL2 k = new DoubleGaussL2();
		k.setGamma(0.05);
		LaSVM<double[]> svm = new LaSVM<double[]>(k);
		svm.setC(10);
		OneAgainstAll<double[]> multisvm = new OneAgainstAll<double[]>(svm);
		
		multisvm.train(train);
		
		for(TrainingSample<double[]> t : test) {
			double v = multisvm.valueOf(t.sample);
			if(v != t.label) {
				debug.println(0,  "error : got "+v+" expected "+t.label);
				return false;
			}
		}
		return true;
	}
	
}
