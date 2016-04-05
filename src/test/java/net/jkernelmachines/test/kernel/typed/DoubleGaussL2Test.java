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
package net.jkernelmachines.test.kernel.typed;

import static org.junit.Assert.assertEquals;
import net.jkernelmachines.kernel.typed.DoubleGaussL2;

import org.junit.Before;
import org.junit.Test;

/**
 * @author picard
 *
 */
public class DoubleGaussL2Test {

	DoubleGaussL2 gaussl2;
	
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		gaussl2 = new DoubleGaussL2();
	}

	/**
	 * Test method for {@link net.jkernelmachines.kernel.typed.DoubleGaussL2#setGamma(double)}.
	 */
	@Test
	public final void testSetGamma() {
		gaussl2.setGamma(1.0);
		assertEquals(gaussl2.getGamma(), 1.0, 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.kernel.typed.DoubleGaussL2#DoubleGaussL2(double)}.
	 */
	@Test
	public final void testDoubleGaussL2Double() {
		gaussl2 = new DoubleGaussL2(1.0);
		assertEquals(gaussl2.getGamma(), 1.0, 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.kernel.typed.DoubleGaussL2#valueOf(double[], double[])}.
	 */
	@Test
	public final void testValueOfDoubleArrayDoubleArray() {

		double[] x1 = { 1.0, 0.0};
		double[] x2 = { 0.0, 1.0};
		
		assertEquals(1.0, gaussl2.valueOf(x1, x1), 1e-15);

		gaussl2.setGamma(1000);
		assertEquals(0.0, gaussl2.valueOf(x1, x2), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.kernel.typed.DoubleGaussL2#valueOf(double[])}.
	 */
	@Test
	public final void testValueOfDoubleArray() {
		double[] x1 = { 1.0, 0.0};
		
		assertEquals(1.0, gaussl2.valueOf(x1, x1), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.kernel.typed.DoubleGaussL2#distanceValueOf(double[], double[])}.
	 */
	@Test
	public final void testDistanceValueOfDoubleArrayDoubleArray() {

		double[] x1 = { 1.0, 0.0};
		double[] x2 = { 0.0, 1.0};
		
		assertEquals(0.0, gaussl2.distanceValueOf(x1, x1), 1e-15);

		assertEquals(2.0, gaussl2.distanceValueOf(x1, x2), 1e-15);
	}

}
