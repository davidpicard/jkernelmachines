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
package net.jkernelmachines.test.projection;

import static org.junit.Assert.assertEquals;

import java.util.List;

import net.jkernelmachines.kernel.typed.DoubleGaussL2;
import net.jkernelmachines.kernel.typed.DoubleLinear;
import net.jkernelmachines.projection.KernelPCA;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.generators.GaussianGenerator;

import org.junit.Before;
import org.junit.Test;

/**
 * @author picard
 *
 */
public class KernelPCATest {


	private List<TrainingSample<double[]>> list;
	private KernelPCA<double[]> pca;
	DoubleGaussL2 k;
	

	
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		
		int dim = 16;
		
		GaussianGenerator gen = new GaussianGenerator(dim, 0, 1.0);
		list = gen.generateList(dim);
		
		k = new DoubleGaussL2(2.0);
		pca = new KernelPCA<double[]>(k);
		
		pca.train(list);
	}

	/**
	 * Test method for {@link net.jkernelmachines.projection.KernelPCA#projectList(java.util.List)}.
	 */
	@Test
	public final void testProjectListListOfTrainingSampleOfT() {
		double[][] m1 = k.getKernelMatrix(list);
		
		List<TrainingSample<double[]>> plist = pca.projectList(list);
		DoubleLinear lin = new DoubleLinear();
		double[][] m2 = lin.getKernelMatrix(plist);
		
		for(int i = 0 ; i < m1.length ; i++) {
			for(int j = i ; j < m1[0].length ; j++) {
				assertEquals(m1[i][j], m2[i][j]+pca.getMean(), 1e-10);
			}
		}

	}

}
