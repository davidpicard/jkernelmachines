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
package net.jkernelmachines.example;

import java.io.IOException;
import java.util.List;

import net.jkernelmachines.classifier.LaSVM;
import net.jkernelmachines.evaluation.ApEvaluator;
import net.jkernelmachines.io.LibSVMImporter;
import net.jkernelmachines.kernel.typed.DoubleGaussL2;
import net.jkernelmachines.type.TrainingSample;

/**
 * <p>This class is a very simple introduction to the use of jkernelmachines.</p>
 * 
 * <p>It reads data from a file in libsvm format, then trains a SVM classifier using
 * LaSVM algorithm with a Gaussian kernel. Finally, output values are computed on some 
 * samples and error rate is computed.</p>
 * 
 * @author picard
 *
 */
public class SVMExample {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		//checking arguments
		if(args.length < 1) {
			System.out.println("Usage: SVMExample trainfile testfile");
			return;
		}
		
		//parsing samples from a file using libsvm format
		List<TrainingSample<double[]>> train = null;
		List<TrainingSample<double[]>> test = null;
		try {
			train = LibSVMImporter.importFromFile(args[0]);
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+args[0]);
			return;
		}
		try {
			test = LibSVMImporter.importFromFile(args[1]);
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+args[1]);
			return;
		}
		
		//setting kernel
		DoubleGaussL2 kernel = new DoubleGaussL2();
		kernel.setGamma(2.0);
		
		//setting SVM parameters
		LaSVM<double[]> svm = new LaSVM<double[]>(kernel);
		svm.setC(10); //C hyperparameter
				
		//evaluation on testing set using evaluator
		ApEvaluator<double[]> ape = new ApEvaluator<double[]>();
		ape.setClassifier(svm);
		ape.setTrainingSet(train);
		ape.setTestingSet(test);
		ape.evaluate(); // training and evaluating at the same time
		
		// printing average precision obtained
		System.out.println("Average Precision: "+ape.getScore());
	}

}
