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

import static net.jkernelmachines.util.algebra.VectorOperations.add;
import static net.jkernelmachines.util.algebra.VectorOperations.addi;
import static net.jkernelmachines.util.algebra.VectorOperations.dot;
import static net.jkernelmachines.util.algebra.VectorOperations.mul;
import static net.jkernelmachines.util.algebra.VectorOperations.muli;
import static net.jkernelmachines.util.algebra.VectorOperations.n2;
import static net.jkernelmachines.util.algebra.VectorOperations.n2p2;
import static org.junit.Assert.assertEquals;

import org.junit.Test;

/**
 * @author picard
 *
 */
public class VectorOperationsTest {

	/**
	 * Test method for {@link net.jkernelmachines.util.algebra.VectorOperations#add(double[], double, double[])}.
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
	 * Test method for {@link net.jkernelmachines.util.algebra.VectorOperations#addi(double[], double[], double, double[])}.
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
	 * Test method for {@link net.jkernelmachines.util.algebra.VectorOperations#mul(double[], double)}.
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
	 * Test method for {@link net.jkernelmachines.util.algebra.VectorOperations#muli(double[], double[], double)}.
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
	 * Test method for {@link net.jkernelmachines.util.algebra.VectorOperations#dot(double[], double[])}.
	 */
	@Test
	public final void testDot() {
		double[] A = {1, 2};
		double[] B = {3, 4};
		
		double d = dot(A, B);
		
		assertEquals(11, d, 1e-15);	
	}

	/**
	 * Test method for {@link net.jkernelmachines.util.algebra.VectorOperations#n2(double[])}.
	 */
	@Test
	public final void testN2() {
		double[] A = {1, 2};
		
		double n = dot(A, A);
		double n2 = n2(A);
		
		assertEquals(Math.sqrt(n), n2, 1e-15);	
	}

	/**
	 * Test method for {@link net.jkernelmachines.util.algebra.VectorOperations#n2p2(double[])}.
	 */
	@Test
	public final void testN2p2() {
		double[] A = {1, 2};
		
		double n = dot(A, A);
		double n2 = n2p2(A);
		
		assertEquals(n, n2, 1e-15);	
	}

}
