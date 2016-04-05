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
package net.jkernelmachines.test.util.generators;

import static org.junit.Assert.assertEquals;

import java.util.List;

import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.generators.MultiClassGaussianGenerator;

import org.junit.Before;
import org.junit.Test;

/**
 * @author picard
 *
 */
public class MultiClassGaussianGeneratorTest {

	MultiClassGaussianGenerator mcg;
	
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		mcg = new MultiClassGaussianGenerator();
	}

	/**
	 * Test method for {@link net.jkernelmachines.util.generators.MultiClassGaussianGenerator#MultiClassGaussianGenerator(int)}.
	 */
	@Test
	public final void testMultiClassGaussianGeneratorInt() {
		mcg = new MultiClassGaussianGenerator(5);
		assertEquals(5, mcg.getNbclasses());
	}

	/**
	 * Test method for {@link net.jkernelmachines.util.generators.MultiClassGaussianGenerator#generateList(int)}.
	 */
	@Test
	public final void testGenerateList() {
		mcg.setNbclasses(4);
		int[] nbSamples = {0, 0, 0, 0};
		
		List<TrainingSample<double[]>> l = mcg.generateList(10);
		
		for(TrainingSample<double[]> t : l)
			nbSamples[t.label]++;
		for(int i = 0 ; i < 4 ; i++)
		assertEquals(10, nbSamples[i]);
	}

	/**
	 * Test method for {@link net.jkernelmachines.util.generators.MultiClassGaussianGenerator#setP(float)}.
	 */
	@Test
	public final void testSetP() {
		mcg.setP(2.0f);
		assertEquals(2.0f, mcg.getP(), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.util.generators.MultiClassGaussianGenerator#setSigma(double)}.
	 */
	@Test
	public final void testSetSigma() {
		mcg.setSigma(1.0);
		assertEquals(1.0, mcg.getSigma(), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.util.generators.MultiClassGaussianGenerator#setNbclasses(int)}.
	 */
	@Test
	public final void testSetNbclasses() {
		mcg.setNbclasses(10);
		assertEquals(10, mcg.getNbclasses());
	}

}
