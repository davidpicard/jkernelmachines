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
import net.jkernelmachines.util.generators.GaussianGenerator;

import org.junit.Before;
import org.junit.Test;

/**
 * @author picard
 *
 */
public class GaussianGeneratorTest {
	
	GaussianGenerator g;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		g = new GaussianGenerator();
	}

	/**
	 * Test method for {@link net.jkernelmachines.util.generators.GaussianGenerator#GaussianGenerator(int)}.
	 */
	@Test
	public final void testGaussianGeneratorInt() {
		g = new GaussianGenerator(2);
		assertEquals(2, g.getDimension());
	}

	/**
	 * Test method for {@link net.jkernelmachines.util.generators.GaussianGenerator#GaussianGenerator(int, float, double)}.
	 */
	@Test
	public final void testGaussianGeneratorIntFloatDouble() {
		g = new GaussianGenerator(2, 4.0f, 1.0);
		assertEquals(2, g.getDimension());
		assertEquals(4.0f, g.getP(), 1e-15);
		assertEquals(1.0, g.getSigma(), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.util.generators.GaussianGenerator#generateList(int)}.
	 */
	@Test
	public final void testGenerateListInt() {
		List<TrainingSample<double[]>> l = g.generateList(10);
		assertEquals(10, l.size());
	}

	/**
	 * Test method for {@link net.jkernelmachines.util.generators.GaussianGenerator#generateList(int, int)}.
	 */
	@Test
	public final void testGenerateListIntInt() {
		List<TrainingSample<double[]>> l = g.generateList(10,  10);
		int nbpos = 0, nbneg = 0;
		for(TrainingSample<double[]> t : l)
			if(t.label > 0)
				nbpos++;
			else if( t.label < 0)
				nbneg++;
		
		assertEquals(10, nbpos);
		assertEquals(10, nbneg);
	}

	/**
	 * Test method for {@link net.jkernelmachines.util.generators.GaussianGenerator#setP(float)}.
	 */
	@Test
	public final void testSetP() {
		g.setP(4.0f);
		assertEquals(4.0f, g.getP(), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.util.generators.GaussianGenerator#setSigma(double)}.
	 */
	@Test
	public final void testSetSigma() {
		g.setSigma(1.0);
		assertEquals(1.0, g.getSigma(), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.util.generators.GaussianGenerator#setDimension(int)}.
	 */
	@Test
	public final void testSetDimension() {
		g.setDimension(2);
		assertEquals(2, g.getDimension());
	}

}
