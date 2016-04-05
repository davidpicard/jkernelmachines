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
package net.jkernelmachines.test.classifier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.List;

import net.jkernelmachines.classifier.TSMKL;
import net.jkernelmachines.kernel.typed.DoubleGaussL2;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.generators.GaussianGenerator;

import org.junit.Before;
import org.junit.Test;

/**
 * @author picard
 *
 */
public class TSMKLTest {

	List<TrainingSample<double[]>> train;
	TSMKL<double[]> svm;
	
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		
		GaussianGenerator g = new GaussianGenerator(10, 5.0f, 1.0);
		train = g.generateList(10);
		
		svm = new TSMKL<double[]>();
		svm.addKernel(new DoubleGaussL2());
	}

	/**
	 * Test method for {@link net.jkernelmachines.classifier.TSMKL#train(java.util.List)}.
	 */
	@Test
	public final void testTrainListOfTrainingSampleOfT() {
		svm.train(train);
		for(TrainingSample<double[]> t : train) {
			double v = t.label * svm.valueOf(t.sample);
			assertTrue(v > 0);
		}
	}

	/**
	 * Test method for {@link net.jkernelmachines.classifier.TSMKL#addKernel(net.jkernelmachines.kernel.Kernel)}.
	 */
	@Test
	public final void testAddKernel() {
		DoubleGaussL2 k = new DoubleGaussL2(1.0);
		svm.addKernel(k);
		assertTrue(svm.getKernels().contains(k));
	}


	/**
	 * Test method for {@link net.jkernelmachines.classifier.TSMKL#setC(double)}.
	 */
	@Test
	public final void testSetC() {
		svm.setC(10.0);
		assertEquals(10.0, svm.getC(), 1e-15);
	}

}
