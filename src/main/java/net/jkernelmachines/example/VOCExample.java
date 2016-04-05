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

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.List;

import net.jkernelmachines.classifier.DoubleSAG;
import net.jkernelmachines.evaluation.ApEvaluator;
import net.jkernelmachines.io.FvecImporter;
import net.jkernelmachines.type.TrainingSample;

/**
 * Simple program to compute the average precision for the PASCAL VOC challenge.
 * Data has to be in fvec file format, and groundtruth in the original VOC
 * format
 * 
 * @author picard
 * 
 */
public class VOCExample {

	/**
	 * Simple program to compute the average precision for the PASCAL VOC
	 * challenge. Data has to be in fvec file format, and groundtruth in the
	 * original VOC format
	 * 
	 * @param args
	 *            train_gt.voc train_data.fvec test_gt.txt test_data.fvec
	 */
	public static void main(String[] args) {

		if (args.length < 4) {
			System.out
					.println("usage: VOCExample trainfile.voc trainfile.fvec testfile.voc testfile.fvec");
			return;
		}

		FvecImporter fvecimp = new FvecImporter();

		// read train features
		List<double[]> feat = null;
		try {
			feat = fvecimp.readFile(args[1]);
		} catch (IOException e1) {
			System.out.println("Unable to read train features: " + args[1]);
			return;
		}

		// read voc train file
		List<TrainingSample<double[]>> train = new ArrayList<TrainingSample<double[]>>();
		try {
			LineNumberReader lin = new LineNumberReader(new FileReader(args[0]));
			String line;
			int i = 0;
			while ((line = lin.readLine()) != null) {
				// get label from second field. ex: "000012 -1"
				int label = Integer.parseInt(line.split("[ ]+")[1]);
				train.add(new TrainingSample<double[]>(feat.get(i), label));
				i++;
			}
			lin.close();
		} catch (FileNotFoundException e) {
			System.out
					.println("trainfile.voc : " + args[0] + " was not found.");
			return;
		} catch (IOException e) {
			System.out
					.println("Error while parsing trainfile.voc : " + args[0]);
			return;
		}
		System.out.println("Train features loaded.");

		// load test features
		try {
			feat = fvecimp.readFile(args[3]);
		} catch (IOException e1) {
			System.out.println("Unable to read test features: " + args[3]);
			return;
		}

		// read voc test file
		List<TrainingSample<double[]>> test = new ArrayList<TrainingSample<double[]>>();
		try {
			LineNumberReader lin = new LineNumberReader(new FileReader(args[2]));
			String line;
			int i = 0;
			while ((line = lin.readLine()) != null) {
				// get label from second field. ex: "000012 -1"
				int label = Integer.parseInt(line.split("[ ]+")[1]);
				test.add(new TrainingSample<double[]>(feat.get(i), label));
				i++;
			}
			lin.close();
		} catch (FileNotFoundException e) {
			System.out
					.println("trainfile.voc : " + args[2] + " was not found.");
			return;
		} catch (IOException e) {
			System.out
					.println("Error while parsing trainfile.voc : " + args[2]);
			return;
		}
		System.out.println("Test features loaded.");

		// classifier
		DoubleSAG svm = new DoubleSAG();
		svm.setE(10);

		// AP evaluation
		ApEvaluator<double[]> ape = new ApEvaluator<double[]>(svm, train, test);
		System.out.println("training...");
		ape.evaluate();
		System.out.println("AP: " + ape.getScore());
	}

}
