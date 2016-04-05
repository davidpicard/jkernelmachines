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
