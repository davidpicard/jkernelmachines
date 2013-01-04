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
package fr.lip6.jkernelmachines.test.util.generators;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.util.List;

import org.junit.Before;
import org.junit.Test;

import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.generators.MultiClassGaussianGenerator;

/**
 * @author picard
 *
 */
public class MultiClassGaussianGeneratorTest {

	MultiClassGaussianGenerator mcg;
	
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		mcg = new MultiClassGaussianGenerator();
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.generators.MultiClassGaussianGenerator#MultiClassGaussianGenerator(int)}.
	 */
	@Test
	public final void testMultiClassGaussianGeneratorInt() {
		mcg = new MultiClassGaussianGenerator(5);
		assertEquals(5, mcg.getNbclasses());
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.generators.MultiClassGaussianGenerator#generateList(int)}.
	 */
	@Test
	public final void testGenerateList() {
		mcg.setNbclasses(4);
		int[] nbSamples = {0, 0, 0, 0};
		
		List<TrainingSample<double[]>> l = mcg.generateList(10);
		
		for(TrainingSample<double[]> t : l)
			nbSamples[t.label]++;
		for(int i = 0 ; i < 4 ; i++)
		assertEquals(10, nbSamples[i]);
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.generators.MultiClassGaussianGenerator#setP(float)}.
	 */
	@Test
	public final void testSetP() {
		mcg.setP(2.0f);
		assertEquals(2.0f, mcg.getP(), 1e-15);
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.generators.MultiClassGaussianGenerator#setSigma(double)}.
	 */
	@Test
	public final void testSetSigma() {
		mcg.setSigma(1.0);
		assertEquals(1.0, mcg.getSigma(), 1e-15);
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.generators.MultiClassGaussianGenerator#setNbclasses(int)}.
	 */
	@Test
	public final void testSetNbclasses() {
		mcg.setNbclasses(10);
		assertEquals(10, mcg.getNbclasses());
	}

}
