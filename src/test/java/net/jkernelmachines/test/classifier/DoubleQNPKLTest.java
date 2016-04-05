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
package net.jkernelmachines.test.classifier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.util.List;

import net.jkernelmachines.classifier.DoubleQNPKL;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.generators.GaussianGenerator;

import org.junit.Before;
import org.junit.Test;

/**
 * @author picard
 *
 */
public class DoubleQNPKLTest {
	
	List<TrainingSample<double[]>> train;
	DoubleQNPKL svm;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		
		GaussianGenerator g = new GaussianGenerator(10, 5.0f, 1.0);
		train = g.generateList(10);
		
		svm = new DoubleQNPKL();
	}

	/**
	 * Test method for {@link net.jkernelmachines.classifier.DoubleQNPKL#train(java.util.List)}.
	 */
	@Test
	public final void testTrainListOfTrainingSampleOfdouble() {
		svm.train(train);
		for(TrainingSample<double[]> t : train) {
			double v = t.label * svm.valueOf(t.sample);
			assertTrue(v > 0);
		}
	}

	/**
	 * Test method for {@link net.jkernelmachines.classifier.DoubleQNPKL#setC(double)}.
	 */
	@Test
	public final void testSetC() {
		svm.setC(1.0);
		assertEquals(1.0, svm.getC(), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.classifier.DoubleQNPKL#setPNorm(double)}.
	 */
	@Test
	public final void testSetPNorm() {
		svm.setPNorm(2.0);
		assertEquals(2.0, svm.getPNorm(), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.classifier.DoubleQNPKL#setStopGap(double)}.
	 */
	@Test
	public final void testSetStopGap() {
		svm.setStopGap(1e-3);
		assertEquals(1e-3, svm.getStopGap(), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.classifier.DoubleQNPKL#setHasNorm(boolean)}.
	 */
	@Test
	public final void testSetHasNorm() {
		svm.setHasNorm(true);
		assertTrue(svm.isHasNorm());
	}

}
