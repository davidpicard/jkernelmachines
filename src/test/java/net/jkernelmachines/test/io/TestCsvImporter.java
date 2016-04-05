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
