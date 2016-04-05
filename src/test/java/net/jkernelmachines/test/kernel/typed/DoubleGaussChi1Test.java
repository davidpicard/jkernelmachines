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
package net.jkernelmachines.test.kernel.typed;

import static org.junit.Assert.assertEquals;
import net.jkernelmachines.kernel.typed.DoubleGaussChi1;

import org.junit.Before;
import org.junit.Test;

/**
 * @author picard
 *
 */
public class DoubleGaussChi1Test {
	
	DoubleGaussChi1 gausschi1;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		gausschi1 = new DoubleGaussChi1();
	}

	/**
	 * Test method for {@link net.jkernelmachines.kernel.typed.DoubleGaussChi1#setGamma(double)}.
	 */
	@Test
	public final void testSetGamma() {
		gausschi1.setGamma(1.0);
		assertEquals(1.0, gausschi1.getGamma(), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.kernel.typed.DoubleGaussChi1#DoubleGaussChi2(double)}.
	 */
	@Test
	public final void testDoubleGaussChi2Double() {
		gausschi1 = new DoubleGaussChi1(1.0);
		assertEquals(1.0, gausschi1.getGamma(), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.kernel.typed.DoubleGaussChi1#valueOf(double[], double[])}.
	 */
	@Test
	public final void testValueOfDoubleArrayDoubleArray() {
		double[] x1 = {1.0, 0.0};
		double[] x2 = {0.0, 1.0};
		
		assertEquals(1.0, gausschi1.valueOf(x1, x1), 1e-15);
		gausschi1.setGamma(1000);
		assertEquals(0.0, gausschi1.valueOf(x1, x2), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.kernel.typed.DoubleGaussChi1#valueOf(double[])}.
	 */
	@Test
	public final void testValueOfDoubleArray() {
		double[] x1 = { 1.0, 0.0 };
		assertEquals(1.0, gausschi1.valueOf(x1), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.kernel.typed.DoubleGaussChi1#distanceValueOf(double[], double[])}.
	 */
	@Test
	public final void testDistanceValueOfDoubleArrayDoubleArray() {
		double[] x1 = {1.0, 0.0};
		double[] x2 = {0.0, 1.0};
		
		assertEquals(0.0, gausschi1.distanceValueOf(x1, x1), 1e-15);
		assertEquals(2.0, gausschi1.distanceValueOf(x1, x2), 1e-15);
	}

}
