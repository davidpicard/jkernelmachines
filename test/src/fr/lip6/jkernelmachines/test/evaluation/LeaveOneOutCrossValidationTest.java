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
package fr.lip6.jkernelmachines.test.evaluation;

import static org.junit.Assert.*;

import java.util.List;

import org.junit.Before;
import org.junit.Test;

import fr.lip6.jkernelmachines.classifier.LaSVM;
import fr.lip6.jkernelmachines.evaluation.AccuracyEvaluator;
import fr.lip6.jkernelmachines.evaluation.LeaveOneOutCrossValidation;
import fr.lip6.jkernelmachines.evaluation.RandomSplitCrossValidation;
import fr.lip6.jkernelmachines.kernel.typed.DoubleGaussL2;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.generators.GaussianGenerator;

/**
 * @author picard
 *
 */
public class LeaveOneOutCrossValidationTest {
	List<TrainingSample<double[]>> train;
	LaSVM<double[]> svm;

	@Before
	public void setUp() throws Exception {
		
		GaussianGenerator g = new GaussianGenerator(10, 5.0f, 0.1);
		train = g.generateList(50);
		
		DoubleGaussL2 k = new DoubleGaussL2(1.0);
		svm = new LaSVM<double[]>(k);
		svm.setC(1.0);
	}


	/**
	 * Test method for {@link fr.lip6.jkernelmachines.evaluation.LeaveOneOutCrossValidation#run()}.
	 */
	@Test
	public final void testRun() {
		AccuracyEvaluator<double[]> ae = new AccuracyEvaluator<double[]>();
		LeaveOneOutCrossValidation<double[]> rscv = new LeaveOneOutCrossValidation<double[]>(svm, train, ae);
		
		rscv.run();
		
		assertEquals(1.0, rscv.getAverageScore(), 1e-15);
	}

}
