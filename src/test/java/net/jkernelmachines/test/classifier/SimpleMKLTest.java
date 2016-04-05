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

import net.jkernelmachines.classifier.LaSVM;
import net.jkernelmachines.classifier.SimpleMKL;
import net.jkernelmachines.kernel.typed.DoubleGaussL2;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.generators.GaussianGenerator;

import org.junit.Before;
import org.junit.Test;

/**
 * @author picard
 *
 */
public class SimpleMKLTest {

	List<TrainingSample<double[]>> train;
	SimpleMKL<double[]> svm;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		
		GaussianGenerator g = new GaussianGenerator(10, 5.0f, 1.0);
		train = g.generateList(10);
		
		svm = new SimpleMKL<double[]>();
		svm.addKernel(new DoubleGaussL2());
	}

	/**
	 * Test method for {@link net.jkernelmachines.classifier.SimpleMKL#addKernel(net.jkernelmachines.kernel.Kernel)}.
	 */
	@Test
	public final void testAddKernel() {
		DoubleGaussL2 k = new DoubleGaussL2(1.0);
		svm.addKernel(k);
		assertTrue(svm.getKernels().contains(k));
	}

	/**
	 * Test method for {@link net.jkernelmachines.classifier.SimpleMKL#train(java.util.List)}.
	 */
	@Test
	public final void testTrainListOfTrainingSampleOfT() {
		svm.train(train);
		for(TrainingSample<double[]> t : train) {
			double v = t.label * svm.valueOf(t.sample);
			assertTrue(v > 0);
		}
	}

	/**
	 * Test method for {@link net.jkernelmachines.classifier.SimpleMKL#setC(double)}.
	 */
	@Test
	public final void testSetC() {
		svm.setC(10.0);
		assertEquals(10.0, svm.getC(), 1e-15);
	}

	/**
	 * Test method for {@link net.jkernelmachines.classifier.SimpleMKL#setClassifier(net.jkernelmachines.classifier.KernelSVM)}.
	 */
	@Test
	public final void testSetClassifier() {
		LaSVM<double[]> lasvm = new LaSVM<double[]>(new DoubleGaussL2());
		svm.setClassifier(lasvm);
		assertEquals(lasvm, svm.getClassifier());
	}

}
