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
package net.jkernelmachines.test.kernel.extra;

import static org.junit.Assert.assertEquals;

import java.util.List;

import net.jkernelmachines.kernel.extra.NystromKernel;
import net.jkernelmachines.kernel.typed.DoubleGaussL2;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.generators.GaussianGenerator;

import org.junit.Before;
import org.junit.Test;

/**
 * Test case for the NystromKernel class
 * @author picard
 *
 */
public class NystromKernelTest {
	
	List<TrainingSample<double[]>> list;
	DoubleGaussL2 dk;
	NystromKernel<double[]> nk;
	int nb_samples = 50;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		GaussianGenerator g = new GaussianGenerator(2, 1, 0.5);
		list = g.generateList(nb_samples);
		dk = new DoubleGaussL2();
		nk = new NystromKernel<double[]>(dk);
	}

	/**
	 * Test method for {@link net.jkernelmachines.kernel.extra.NystromKernel#train(java.util.List)}.
	 */
	@Test
	public final void testTrain() {
		double[][] matrix = dk.getKernelMatrix(list);
		nk.train(list);
		double[][] nk_matrix = nk.getKernelMatrix(list);
		
		for(int i = 0 ; i < matrix.length ; i++) {
			for(int j = i ; j< matrix.length ; j++) {
				assertEquals(matrix[i][j], nk_matrix[i][j], 1e-10);
			}
		}
	}

	/**
	 * Test method for {@link net.jkernelmachines.kernel.extra.NystromKernel#activeTrain(java.util.List, int, int, int)}.
	 */
	@Test
	public final void testActiveTrain() {
		double[][] matrix = dk.getKernelMatrix(list);
		nk.activeTrain(list, nb_samples, 1, 5);
		double[][] nk_matrix = nk.getKernelMatrix(list);
		
		for(int i = 0 ; i < matrix.length ; i++) {
			for(int j = i ; j< matrix.length ; j++) {
				assertEquals(matrix[i][j], nk_matrix[i][j], 1e-10);
			}
		}
	}

}
