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

    Copyright David Picard - 2013

 */
package fr.lip6.jkernelmachines.example;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.List;

import fr.lip6.jkernelmachines.classifier.DoubleSAG;
import fr.lip6.jkernelmachines.evaluation.ApEvaluator;
import fr.lip6.jkernelmachines.io.FvecImporter;
import fr.lip6.jkernelmachines.type.TrainingSample;

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
