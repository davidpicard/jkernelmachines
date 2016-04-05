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
package net.jkernelmachines.test.io;

import java.io.IOException;
import java.util.Collections;
import java.util.List;

import net.jkernelmachines.classifier.LaSVM;
import net.jkernelmachines.evaluation.AccuracyEvaluator;
import net.jkernelmachines.evaluation.NFoldCrossValidation;
import net.jkernelmachines.io.CsvImporter;
import net.jkernelmachines.kernel.typed.DoubleGaussL2;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.DataPreProcessing;

/**
 * Train a SVM using LaSVM algorithm on data in the csv format
 * 
 * usage: TestCsvImporter trainfile testfile
 * @author picard
 *
 */
public class TestCsvImporter {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		if(args.length < 2) {
			System.out.println("Usage: TestCsvImporter trainfile testfile");
			return;
		}
		
		List<TrainingSample<double[]>> trainlist = null;
		
		//parsing
		try {
			trainlist = CsvImporter.importFromFile(args[0]);
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+args[0]);
			return;
		}
		System.out.println(trainlist.size()+" training data");
		System.out.println("dimension of samples: "+trainlist.get(0).sample.length);
		
		Collections.shuffle(trainlist);
		DataPreProcessing.centerList(trainlist);
		DataPreProcessing.reduceList(trainlist);
		DataPreProcessing.normalizeList(trainlist);
		
		//learning
		DoubleGaussL2 kernel = new DoubleGaussL2();
		kernel.setGamma(2.0);
		LaSVM<double[]> svm = new LaSVM<double[]>(kernel);
		svm.setC(10);
		svm.setE(5);
				
		AccuracyEvaluator<double[]> accev = new AccuracyEvaluator<double[]>();
		NFoldCrossValidation<double[]> cv = new NFoldCrossValidation<double[]>(5, svm, trainlist, accev);
		
		cv.run();
		
		System.out.println("accuracy: "+cv.getAverageScore()+" +/- "+cv.getStdDevScore());

	}

}
