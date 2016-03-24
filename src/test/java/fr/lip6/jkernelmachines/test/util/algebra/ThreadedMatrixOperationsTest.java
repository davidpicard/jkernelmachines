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

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import fr.lip6.jkernelmachines.util.algebra.ThreadedMatrixOperations;

/**
 * @author picard
 *
 */
public class ThreadedMatrixOperationsTest {



	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.algebra.ThreadedMatrixOperations#trans(double[][])}.
	 */
	@Test
	public final void testTrans() {
		double[][] A = {{1, 2}, {3, 4}};
		
		ThreadedMatrixOperations.granularity = 0;
		double[][] B = ThreadedMatrixOperations.trans(A);
		assertEquals(1, B[0][0], 1e-15);
		assertEquals(3, B[0][1], 1e-15);
		assertEquals(2, B[1][0], 1e-15);
		assertEquals(4, B[1][1], 1e-15);
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.algebra.ThreadedMatrixOperations#transi(double[][])}.
	 */
	@Test
	public final void testTransi() {
		double[][] A = {{1, 2}, {3, 4}};
		
		ThreadedMatrixOperations.granularity = 0;
		ThreadedMatrixOperations.transi(A);
		assertEquals(1, A[0][0], 1e-15);
		assertEquals(3, A[0][1], 1e-15);
		assertEquals(2, A[1][0], 1e-15);
		assertEquals(4, A[1][1], 1e-15);
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.algebra.ThreadedMatrixOperations#mul(double[][], double[][])}.
	 */
	@Test
	public final void testMul() {
		double[][] A = {{2, 1}, {3, 0}, {4, 0}};
		double[][] B = {{0, 1, 0}, {1000, 100, 10}};
		
		ThreadedMatrixOperations.granularity = 0;
		double[][] C = ThreadedMatrixOperations.mul(A, B);
		
		assertEquals(1000, C[0][0], 1e-15);
		assertEquals(102, C[0][1], 1e-15);
		assertEquals(10, C[0][2], 1e-15);
		assertEquals(0, C[1][0], 1e-15);
		assertEquals(3, C[1][1], 1e-15);
		assertEquals(0, C[1][2], 1e-15);
		assertEquals(0, C[2][0], 1e-15);
		assertEquals(4, C[2][1], 1e-15);
		assertEquals(0, C[2][2], 1e-15);
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.algebra.ThreadedMatrixOperations#muli(double[][], double[][], double[][])}.
	 */
	@Test
	public final void testMuli() {
		double[][] A = {{2, 1}, {3, 0}, {4, 0}};
		double[][] B = {{0, 1, 0}, {1000, 100, 10}};
		
		double[][] C = new double[3][3];
		
		ThreadedMatrixOperations.granularity = 0;
		ThreadedMatrixOperations.muli(C,  A,  B);
		
		assertEquals(1000, C[0][0], 1e-15);
		assertEquals(102, C[0][1], 1e-15);
		assertEquals(10, C[0][2], 1e-15);
		assertEquals(0, C[1][0], 1e-15);
		assertEquals(3, C[1][1], 1e-15);
		assertEquals(0, C[1][2], 1e-15);
		assertEquals(0, C[2][0], 1e-15);
		assertEquals(4, C[2][1], 1e-15);
		assertEquals(0, C[2][2], 1e-15);
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.algebra.ThreadedMatrixOperations#transMul(double[][], double[][])}.
	 */
	@Test
	public final void testTransMul() {
		double[][] A = {{2, 1}, {3, 0}};
		double[][] B = {{0, 1}, {1000, 100}};
		
		ThreadedMatrixOperations.granularity = 0;
		double[][] C = ThreadedMatrixOperations.mul(ThreadedMatrixOperations.trans(A), B);
		
		double[][] Cprime = ThreadedMatrixOperations.transMul(A, B);
		
		for(int i = 0 ; i < 2 ; i++) {
			for(int j = 0 ; j < 2 ; j++) {
				assertEquals(C[i][j], Cprime[i][j], 1e-15);
			}
		}
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.util.algebra.ThreadedMatrixOperations#transMuli(double[][], double[][], double[][])}.
	 */
	@Test
	public final void testTransMuli() {
		double[][] A = {{2, 1}, {3, 0}};
		double[][] B = {{0, 1}, {1000, 100}};
		
		ThreadedMatrixOperations.granularity = 0;
		double[][] C = ThreadedMatrixOperations.mul(ThreadedMatrixOperations.trans(A), B);
		
		double[][] Cprime = new double[2][2];
		ThreadedMatrixOperations.transMuli(Cprime, A, B);
		
		for(int i = 0 ; i < 2 ; i++) {
			for(int j = 0 ; j < 2 ; j++) {
				assertEquals(C[i][j], Cprime[i][j], 1e-15);
			}
		}
	}
	
}
