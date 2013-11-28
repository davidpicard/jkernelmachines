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
package fr.lip6.jkernelmachines.test.classifier;

import static org.junit.Assert.assertTrue;

import java.util.List;

import org.junit.Before;
import org.junit.Test;

import fr.lip6.jkernelmachines.classifier.NystromLSSVM;
import fr.lip6.jkernelmachines.kernel.typed.DoubleGaussL2;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.generators.GaussianGenerator;

/**
 * @author picard
 *
 */
public class NystromLSSVMTest {


	List<TrainingSample<double[]>> train;
	NystromLSSVM<double[]> svm;
	
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {

		GaussianGenerator g = new GaussianGenerator(10, 5.0f, 1.0);
		train = g.generateList(10);
		
		DoubleGaussL2 k = new DoubleGaussL2(1.0);
		svm = new NystromLSSVM<double[]>(k);
		svm.setPercent(0.8);
		svm.setIteration(2);
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.classifier.NystromLSSVM#train(java.util.List)}.
	 */
	@Test
	public final void testTrainListOfTrainingSampleOfT() {
		svm.train(train);
		for(TrainingSample<double[]> t : train) {
			double v = t.label * svm.valueOf(t.sample);
			assertTrue(v > 0);
		}
	}

}
