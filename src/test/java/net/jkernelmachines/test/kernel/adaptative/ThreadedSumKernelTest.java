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
package net.jkernelmachines.test.kernel.adaptative;

import static org.junit.Assert.assertEquals;

import java.util.List;

import net.jkernelmachines.kernel.adaptative.ThreadedSumKernel;
import net.jkernelmachines.kernel.typed.index.IndexDoubleGaussL2;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.generators.GaussianGenerator;

import org.junit.Test;

/**
 * @author picard
 *
 */
public class ThreadedSumKernelTest {

	/**
	 * Test method for {@link net.jkernelmachines.kernel.adaptative.ThreadedSumKernel#getKernelMatrixLine(java.lang.Object, java.util.List)}.
	 */
	@Test
	public final void testGetKernelMatrixLine() {
		
		GaussianGenerator gg = new GaussianGenerator(16);
		List<TrainingSample<double[]>> l = gg.generateList(512);
		
		double[] w = new double[16];
		double sum = 0;
		for(int i = 0 ; i < 16 ; i++) {
			w[i] = Math.random();
			sum += w[i];
		}
		for(int i = 0 ; i < 16 ; i++) {
			w[i] /= sum;
		}
		
		ThreadedSumKernel<double[]> tsk = new ThreadedSumKernel<>();
		for(int i = 0 ; i < 16 ; i++) {
			tsk.addKernel(new IndexDoubleGaussL2(i), w[i]);
		}
		
		double[][] matrix = tsk.getKernelMatrix(l);
		double[][] lines = new double[l.size()][];
		for(int i = 0 ; i < l.size() ; i++) {
			lines[i] = tsk.getKernelMatrixLine(l.get(i).sample, l);
		}
		
		for(int i = 0 ; i < matrix.length ; i++) {
			for(int j = i ; j< matrix.length ; j++) {
				assertEquals(matrix[i][j], lines[i][j], 1e-10);
			}
		}
	}

}
