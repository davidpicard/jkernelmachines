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
package fr.lip6.example;

import java.io.IOException;
import java.util.List;

import fr.lip6.classifier.LaSVM;
import fr.lip6.io.LibSVMImporter;
import fr.lip6.kernel.typed.DoubleGaussL2;
import fr.lip6.type.TrainingSample;

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
			System.out.println("Usage: SVMExample datafile");
			return;
		}
		
		//parsing samples from a file using libsvm format
		List<TrainingSample<double[]>> list = null;
		try {
			list = LibSVMImporter.importFromFile(args[0]);
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+args[0]);
			return;
		}
		
		
		//setting kernel
		DoubleGaussL2 kernel = new DoubleGaussL2();
		kernel.setGamma(2.0);
		
		//setting SVM parameters
		LaSVM<double[]> svm = new LaSVM<double[]>(kernel);
		svm.setC(10); //C hyperparameter
		
		//splitting list into training and testing sets using half the samples
		List<TrainingSample<double[]>> train = list.subList(0, list.size()/2);
		List<TrainingSample<double[]>> test = list.subList(list.size()/2, list.size());
		
		//training the classifier
		svm.train(train);
		
		//evaluation on testing set
		double err = 0;
		for(TrainingSample<double[]> t : test) {
			//using valueOf() method to get the output for one sample
			double v = svm.valueOf(t.sample);
			if( v * t.label < 0) {
				err++;
			}
		}
		
		// printing error rate
		System.out.println("error rate: "+(err/test.size()));
	}

}
