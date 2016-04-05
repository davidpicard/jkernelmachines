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

import net.jkernelmachines.kernel.extra.CustomTrainTestMatrixKernel;
import net.jkernelmachines.kernel.typed.DoubleLinear;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.generators.GaussianGenerator;

import org.junit.Test;

/**
 * @author picard
 *
 */
public class CustomTrainTestMatrixKernelTest {

	/**
	 * Test method for {@link net.jkernelmachines.kernel.extra.CustomTrainTestMatrixKernel#valueOf(java.lang.Integer, java.lang.Integer)}.
	 */
	@Test
	public final void testValueOfIntegerInteger() {
		
		// generate data
		GaussianGenerator gen = new GaussianGenerator(2);
		List<TrainingSample<double[]>> list = gen.generateList(5, 10);
		
		DoubleLinear lin = new DoubleLinear();
		double[][] matrix = lin.getKernelMatrix(list);
		
		double[][] train = new double[5][5];
		double[][] test = new double[10][10];
		
		// fill train
		for(int i = 0 ; i < 5 ; i++) {
			for(int j = 0 ; j < 5 ; j++) {
				train[i][j] = matrix[i][j];
			}
		}
		// fill test
		for(int i = 0 ; i < 5 ; i++) {
			for(int j = 0 ; j < 10 ; j++) {
				test[i][j] = matrix[i][5+j];
			}
		}
		
		// build custom kernel
		CustomTrainTestMatrixKernel kernel = new CustomTrainTestMatrixKernel(train, test);
		
		// assert values from train to end of test
		for(int i = 0 ; i < 5 ; i++) {
			for(int j = 0 ; j < 15 ; j++) {
				assertEquals(matrix[i][j], kernel.valueOf(i, j), 1e-15);
				assertEquals(matrix[j][i], kernel.valueOf(j, i), 1e-15);
			}
		}
		
	}

}
