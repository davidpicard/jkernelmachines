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

import java.util.List;

import org.junit.Before;
import org.junit.Test;

import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.generators.GaussianGenerator;

/**
 * @author picard
 *
 */
public class GaussianGeneratorTest {
	
	GaussianGenerator g;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		g = new GaussianGenerator();
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.generators.GaussianGenerator#GaussianGenerator(int)}.
	 */
	@Test
	public final void testGaussianGeneratorInt() {
		g = new GaussianGenerator(2);
		assertEquals(2, g.getDimension());
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.generators.GaussianGenerator#GaussianGenerator(int, float, double)}.
	 */
	@Test
	public final void testGaussianGeneratorIntFloatDouble() {
		g = new GaussianGenerator(2, 4.0f, 1.0);
		assertEquals(2, g.getDimension());
		assertEquals(4.0f, g.getP(), 1e-15);
		assertEquals(1.0, g.getSigma(), 1e-15);
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.generators.GaussianGenerator#generateList(int)}.
	 */
	@Test
	public final void testGenerateListInt() {
		List<TrainingSample<double[]>> l = g.generateList(10);
		assertEquals(10, l.size());
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.generators.GaussianGenerator#generateList(int, int)}.
	 */
	@Test
	public final void testGenerateListIntInt() {
		List<TrainingSample<double[]>> l = g.generateList(10,  10);
		int nbpos = 0, nbneg = 0;
		for(TrainingSample<double[]> t : l)
			if(t.label > 0)
				nbpos++;
			else if( t.label < 0)
				nbneg++;
		
		assertEquals(10, nbpos);
		assertEquals(10, nbneg);
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.generators.GaussianGenerator#setP(float)}.
	 */
	@Test
	public final void testSetP() {
		g.setP(4.0f);
		assertEquals(4.0f, g.getP(), 1e-15);
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.generators.GaussianGenerator#setSigma(double)}.
	 */
	@Test
	public final void testSetSigma() {
		g.setSigma(1.0);
		assertEquals(1.0, g.getSigma(), 1e-15);
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.generators.GaussianGenerator#setDimension(int)}.
	 */
	@Test
	public final void testSetDimension() {
		g.setDimension(2);
		assertEquals(2, g.getDimension());
	}

}
