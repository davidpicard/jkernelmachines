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

import java.util.List;

import net.jkernelmachines.classifier.LaSVM;
import net.jkernelmachines.classifier.multiclass.OneAgainstAll;
import net.jkernelmachines.evaluation.MulticlassAccuracyEvaluator;
import net.jkernelmachines.evaluation.NFoldCrossValidation;
import net.jkernelmachines.kernel.typed.DoubleGaussL2;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.DebugPrinter;
import net.jkernelmachines.util.generators.MultiClassGaussianGenerator;

/**
 * Example of multiclass classification on artificial dataset.
 * @author picard
 *
 */
public class MulticlassExample {

	/**
	 * Program entry point
	 * @param args ignored
	 */
	public static void main(String[] args) {
		
		//set debug for visibility
		DebugPrinter.DEBUG_LEVEL = 2;
		
		// new generator with 10 classes and stddev of 0.5
		MultiClassGaussianGenerator mcg = new MultiClassGaussianGenerator();
		mcg.setSigma(0.5);
		mcg.setNbclasses(10);
		
		//generate the list
		List<TrainingSample<double[]>> list = mcg.generateList(100);
		
		//build classifier based on GaussL2 lasvm with c=10
		DoubleGaussL2 k =new DoubleGaussL2(2);
		LaSVM<double[]> svm = new LaSVM<double[]>(k);
		svm.setC(10);
		OneAgainstAll<double[]> mcsvm = new OneAgainstAll<double[]>(svm);
		
		
		//doing crossvalidation with multiclass accuracy
		MulticlassAccuracyEvaluator<double[]> eval = new MulticlassAccuracyEvaluator<double[]>();
		NFoldCrossValidation<double[]> cv = new NFoldCrossValidation<double[]>(10, mcsvm, list, eval);
		
		//launch cv
		cv.run();
		
		//print results 
		System.out.println("Multiclass accuracy: "+cv.getAverageScore()+" +/- "+cv.getStdDevScore());

	}

}
