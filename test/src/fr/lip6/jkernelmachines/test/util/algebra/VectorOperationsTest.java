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
package fr.lip6.jkernelmachines.test.util.algebra;

import static fr.lip6.jkernelmachines.util.algebra.VectorOperations.add;
import static fr.lip6.jkernelmachines.util.algebra.VectorOperations.addi;
import static fr.lip6.jkernelmachines.util.algebra.VectorOperations.dot;
import static fr.lip6.jkernelmachines.util.algebra.VectorOperations.mul;
import static fr.lip6.jkernelmachines.util.algebra.VectorOperations.muli;
import static fr.lip6.jkernelmachines.util.algebra.VectorOperations.n2;
import static fr.lip6.jkernelmachines.util.algebra.VectorOperations.n2p2;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

/**
 * @author picard
 *
 */
public class VectorOperationsTest {

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.algebra.VectorOperations#add(double[], double, double[])}.
	 */
	@Test
	public final void testAdd() {
		double[] A = {1, 2};
		double[] B = {3, 4};
		double l = 5;
		
		double[] C = add(A, l, B);
		
		assertEquals(16, C[0], 1e-15);
		assertEquals(22, C[1], 1e-15);		
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.algebra.VectorOperations#addi(double[], double[], double, double[])}.
	 */
	@Test
	public final void testAddi() {
		double[] A = {1, 2};
		double[] B = {3, 4};
		double l = 5;
		
		double[] C = new double[2];
		
		addi(C, A, l, B);
		
		assertEquals(16, C[0], 1e-15);
		assertEquals(22, C[1], 1e-15);	
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.algebra.VectorOperations#mul(double[], double)}.
	 */
	@Test
	public final void testMul() {
		double[] A = {1, 2};
		double l = 5;
		
		double[] C = mul(A, l);
		
		assertEquals(5, C[0], 1e-15);
		assertEquals(10, C[1], 1e-15);	
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.algebra.VectorOperations#muli(double[], double[], double)}.
	 */
	@Test
	public final void testMuli() {
		double[] A = {1, 2};
		double l = 5;
		
		double[] C = new double[2];
		
		muli(C, A, l);
		
		assertEquals(5, C[0], 1e-15);
		assertEquals(10, C[1], 1e-15);	
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.algebra.VectorOperations#dot(double[], double[])}.
	 */
	@Test
	public final void testDot() {
		double[] A = {1, 2};
		double[] B = {3, 4};
		
		double d = dot(A, B);
		
		assertEquals(11, d, 1e-15);	
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.algebra.VectorOperations#n2(double[])}.
	 */
	@Test
	public final void testN2() {
		double[] A = {1, 2};
		
		double n = dot(A, A);
		double n2 = n2(A);
		
		assertEquals(Math.sqrt(n), n2, 1e-15);	
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.algebra.VectorOperations#n2p2(double[])}.
	 */
	@Test
	public final void testN2p2() {
		double[] A = {1, 2};
		
		double n = dot(A, A);
		double n2 = n2p2(A);
		
		assertEquals(n, n2, 1e-15);	
	}

}
