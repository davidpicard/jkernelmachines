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
package net.jkernelmachines.test.density;

import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import net.jkernelmachines.density.SDCADensity;
import net.jkernelmachines.kernel.typed.DoubleGaussL2;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.generators.GaussianGenerator;

import org.junit.Before;
import org.junit.Test;

/**
 * Test case for the SDCADensity class
 * @author picard
 * 
 */
public class SDCADensityTest {

	List<double[]> train;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		GaussianGenerator gen = new GaussianGenerator(2, 2, 1.0);
		List<TrainingSample<double[]>> list = gen.generateList(100, 100);
		train = new ArrayList<double[]>();
		for (TrainingSample<double[]> t : list) {
			train.add(t.sample);
		}
	}

	/**
	 * Test method for
	 * {@link net.jkernelmachines.density.SDCADensity#train(java.lang.Object)}
	 * .
	 */
	@Test
	public final void testTrainT() {
		DoubleGaussL2 k = new DoubleGaussL2();
		SDCADensity<double[]> de = new SDCADensity<double[]>(k);
		de.train(train.get(0));

		for (double[] x : train) {
			assertTrue(!Double.isNaN(de.valueOf(x)));
		}
	}

	/**
	 * Test method for
	 * {@link net.jkernelmachines.density.SDCADensity#train(java.util.List)}
	 * .
	 */
	@Test
	public final void testTrainListOfT() {
		DoubleGaussL2 k = new DoubleGaussL2();
		SDCADensity<double[]> de = new SDCADensity<double[]>(k);
		de.setC(1.);
		de.train(train);
		
		for (double[] x : train) {
			assertTrue(!Double.isNaN(de.valueOf(x)));
		}
	}

}
