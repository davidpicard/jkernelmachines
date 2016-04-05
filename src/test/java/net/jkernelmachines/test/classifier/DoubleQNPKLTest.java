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

import net.jkernelmachines.classifier.DoubleQNPKL;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.generators.GaussianGenerator;

import org.junit.Before;
import org.junit.Test;

/**
 * @author picard
 *
 */
public class DoubleQNPKLTest {
	
	List<TrainingSample<double[]>> train;
	DoubleQNPKL svm;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		
		GaussianGenerator g = new GaussianGenerator(10, 5.0f, 1.0);
		train = g.generateList(10);
		
		svm = new DoubleQNPKL();
	}

	/**
	 * Test method for {@link net.jkernelmachines.classifier.DoubleQNPKL#train(java.util.List)}.
	 */
	@Test
	public final void testTrainListOfTrainingSampleOfdouble() {
		svm.train(train);
		for(TrainingSample<double[]> t : train) {
			double v = t.label * svm.valueOf(t.sample);
			assertTrue(v > 0);
		}
	}

	/**
	 * Test method for {@link net.jkernelmachines.classifier.DoubleQNPKL#setC(double)}.
	 */
	@Test
	public final void testSetC() {
		svm.setC(1.0);
		assertEquals(1.0, svm.getC(), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.classifier.DoubleQNPKL#setPNorm(double)}.
	 */
	@Test
	public final void testSetPNorm() {
		svm.setPNorm(2.0);
		assertEquals(2.0, svm.getPNorm(), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.classifier.DoubleQNPKL#setStopGap(double)}.
	 */
	@Test
	public final void testSetStopGap() {
		svm.setStopGap(1e-3);
		assertEquals(1e-3, svm.getStopGap(), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.classifier.DoubleQNPKL#setHasNorm(boolean)}.
	 */
	@Test
	public final void testSetHasNorm() {
		svm.setHasNorm(true);
		assertTrue(svm.isHasNorm());
	}

}
