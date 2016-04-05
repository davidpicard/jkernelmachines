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
package net.jkernelmachines.test.evaluation;

import static org.junit.Assert.assertEquals;

import java.util.List;

import net.jkernelmachines.classifier.LaSVM;
import net.jkernelmachines.classifier.multiclass.OneAgainstAll;
import net.jkernelmachines.evaluation.MulticlassAccuracyEvaluator;
import net.jkernelmachines.kernel.typed.DoubleGaussL2;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.generators.MultiClassGaussianGenerator;

import org.junit.Before;
import org.junit.Test;

/**
 * @author picard
 *
 */
public class MulticlassAccuracyEvaluatorTest {

	List<TrainingSample<double[]>> train;
	OneAgainstAll<double[]> multisvm;
	
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		MultiClassGaussianGenerator mcgg = new MultiClassGaussianGenerator(4);
		mcgg.setP(10);
		mcgg.setSigma(1);
		train = mcgg.generateList(5);
		
		DoubleGaussL2 k = new DoubleGaussL2();
		k.setGamma(0.5);
		LaSVM<double[]> svm = new LaSVM<double[]>(k);
		svm.setC(10);
		multisvm = new OneAgainstAll<double[]>(svm);
	}

	/**
	 * Test method for {@link net.jkernelmachines.evaluation.MulticlassAccuracyEvaluator#evaluate()}.
	 */
	@Test
	public final void testEvaluate() {
		MulticlassAccuracyEvaluator<double[]> ae = new MulticlassAccuracyEvaluator<double[]>();
		ae.setClassifier(multisvm);
		ae.setTrainingSet(train);
		ae.setTestingSet(train);
		
		ae.evaluate();
		
		assertEquals(1.0, ae.getScore(), 1e-15);	
	}

}
