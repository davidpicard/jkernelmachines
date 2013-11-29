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
package fr.lip6.jkernelmachines.example;

import java.util.List;

import fr.lip6.jkernelmachines.classifier.LaSVM;
import fr.lip6.jkernelmachines.classifier.multiclass.OneAgainstAll;
import fr.lip6.jkernelmachines.evaluation.MulticlassAccuracyEvaluator;
import fr.lip6.jkernelmachines.evaluation.NFoldCrossValidation;
import fr.lip6.jkernelmachines.kernel.typed.DoubleGaussL2;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.DebugPrinter;
import fr.lip6.jkernelmachines.util.generators.MultiClassGaussianGenerator;

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
