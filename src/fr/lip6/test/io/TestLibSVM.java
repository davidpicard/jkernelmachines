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
package fr.lip6.test.io;

import java.io.IOException;
import java.util.List;

import fr.lip6.classifier.LaSVM;
import fr.lip6.io.LibSVMImporter;
import fr.lip6.kernel.typed.DoubleGaussL2;
import fr.lip6.type.TrainingSample;

/**
 * Train a SVM using LaSVM algorithm on data in the libsvm format<br/>
 * 
 * usage: TestLibSVM trainfile testfile
 * @author picard
 *
 */
public class TestLibSVM {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		if(args.length < 2) {
			System.out.println("Usage: TestLibSVM trainfile testfile");
			return;
		}
		
		List<TrainingSample<double[]>> trainlist = null;
		List<TrainingSample<double[]>> testlist = null;
		
		//parsing
		try {
			trainlist = LibSVMImporter.importFromFile(args[0]);
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+args[0]);
			return;
		}
		
		try {
			testlist = LibSVMImporter.importFromFile(args[1]);
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+args[1]);
			return;
		}
		
		//learning
		DoubleGaussL2 kernel = new DoubleGaussL2();
		kernel.setGamma(2.0);
		LaSVM<double[]> svm = new LaSVM<double[]>(kernel);
		svm.setC(10);
		svm.setE(5);
		
		svm.train(trainlist);
		
		//evaluation
		double err = 0;
		for(TrainingSample<double[]> t : testlist) {
			double v = svm.valueOf(t.sample);
			if( v * t.label < 0) {
				err++;
			}
		}
		
		System.out.println("error rate: "+(err/testlist.size()));

	}

}
