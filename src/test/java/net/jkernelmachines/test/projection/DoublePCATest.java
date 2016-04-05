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

import net.jkernelmachines.projection.DoublePCA;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.generators.GaussianGenerator;

import org.junit.Before;
import org.junit.Test;

/**
 * Test methods for PCA
 * 
 * @author picard
 *
 */
public class DoublePCATest {
	
	private List<TrainingSample<double[]>> list;
	private DoublePCA pca;
	
	private int nbSamples = 128;
	private int dim = 64;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		

		GaussianGenerator gen = new GaussianGenerator(dim);
		list = gen.generateList(nbSamples);
		
		
		pca = new DoublePCA();
		pca.train(list);
		
	}

	/**
	 * Test method for {@link net.jkernelmachines.projection.DoublePCA#projectList(java.util.List)}.
	 */
	@Test
	public final void testProjectListListOfTrainingSampleOfdouble() {
		
		List<TrainingSample<double[]>> proj = pca.projectList(list);
		
		// compute cov
		double[][] cov = new double[dim][dim];
		for(int i = 0 ; i < dim ; i++) {
			for(int j = 0 ; j < dim ; j++) {
				for(TrainingSample<double[]> t : proj) {
					cov[i][j] += t.sample[i] * t.sample[j];
				}
				
				cov[i][j] /= list.size();
				if(i!=j) {
					assertEquals(0, cov[i][j], 1e-8);
				}
			}
		}
		
	}

	/**
	 * Test method for {@link net.jkernelmachines.projection.DoublePCA#projectList(java.util.List, boolean)}.
	 */
	@Test
	public final void testProjectListListOfTrainingSampleOfdoubleBoolean() {
		
		List<TrainingSample<double[]>> proj = pca.projectList(list, true);
		
		// compute cov
		double[][] cov = new double[dim][dim];
		for(int i = 0 ; i < dim ; i++) {
			for(int j = 0 ; j < dim ; j++) {
				for(TrainingSample<double[]> t : proj) {
					cov[i][j] += t.sample[i] * t.sample[j];
				}
				cov[i][j] /= list.size();
				if(i!=j) {
					assertEquals(0, cov[i][j], 1e-8);
				}
				else {
					assertEquals(1, cov[i][j], 1e-8);
				}
			}
		}
	}

}
