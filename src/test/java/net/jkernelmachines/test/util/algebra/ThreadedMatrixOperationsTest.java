/*******************************************************************************
 * Copyright (c) 2016, David Picard.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************/
package net.jkernelmachines.test.util.algebra;

import static org.junit.Assert.assertEquals;
import net.jkernelmachines.util.algebra.ThreadedMatrixOperations;

import org.junit.Test;

/**
 * @author picard
 *
 */
public class ThreadedMatrixOperationsTest {



	/**
	 * Test method for {@link net.jkernelmachines.util.algebra.ThreadedMatrixOperations#trans(double[][])}.
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
	 * Test method for {@link net.jkernelmachines.util.algebra.ThreadedMatrixOperations#transi(double[][])}.
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
	 * Test method for {@link net.jkernelmachines.util.algebra.ThreadedMatrixOperations#mul(double[][], double[][])}.
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
	 * Test method for {@link net.jkernelmachines.util.algebra.ThreadedMatrixOperations#muli(double[][], double[][], double[][])}.
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
	 * Test method for {@link net.jkernelmachines.util.algebra.ThreadedMatrixOperations#transMul(double[][], double[][])}.
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
	 * Test method for {@link net.jkernelmachines.util.algebra.ThreadedMatrixOperations#transMuli(double[][], double[][], double[][])}.
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
