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
package fr.lip6.jkernelmachines.test.kernel.typed;

import static org.junit.Assert.assertEquals;

import org.junit.Before;
import org.junit.Test;

import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;

/**
 * @author picard
 *
 */
public class DoubleLinearTest {
	
	private DoubleLinear linear;
	
	@Before
	public void setUp() {
		linear = new DoubleLinear();
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.kernel.typed.DoubleLinear#valueOf(double[], double[])}.
	 */
	@Test
	public final void testValueOfDoubleArrayDoubleArray() {
		
		double[] x1 = { 1.0, 0.0};
		double[] x2 = { 0.0, 1.0};
		
		assertEquals(linear.valueOf(x1, x1), 1.0, 1e-15);
		assertEquals(linear.valueOf(x1, x2), 0.0, 1e-15);
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.kernel.typed.DoubleLinear#valueOf(double[])}.
	 */
	@Test
	public final void testValueOfDoubleArray() {
		double[] x1 = { 1.0, 0.0};
		
		assertEquals(linear.valueOf(x1, x1), 1.0, 1e-15);
	}

}
