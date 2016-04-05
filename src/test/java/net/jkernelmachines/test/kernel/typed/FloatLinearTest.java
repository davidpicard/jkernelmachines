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
import net.jkernelmachines.kernel.typed.FloatLinear;

import org.junit.Before;
import org.junit.Test;

/**
 * @author picard
 *
 */
public class FloatLinearTest {
	
	private FloatLinear linear;
	
	@Before
	public void setUp() {
		linear = new FloatLinear();
	}

	/**
	 * Test method for {@link net.jkernelmachines.kernel.typed.FloatLinear#valueOf(float[], float[])}.
	 */
	@Test
	public final void testValueOfFloatArrayFloatArray() {
		
		float[] x1 = { 1.0f, 0.0f};
		float[] x2 = { 0.0f, 1.0f};
		
		assertEquals(linear.valueOf(x1, x1), 1.0, 1e-15);
		assertEquals(linear.valueOf(x1, x2), 0.0, 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.kernel.typed.FloatLinear#valueOf(float[])}.
	 */
	@Test
	public final void testValueOfFloatArray() {
		float[] x1 = { 1.0f, 0.0f};
		
		assertEquals(linear.valueOf(x1, x1), 1.0, 1e-15);
	}

}
