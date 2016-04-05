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
package net.jkernelmachines.test.evaluation;

import static org.junit.Assert.assertEquals;

import java.util.List;

import net.jkernelmachines.classifier.LaSVM;
import net.jkernelmachines.evaluation.AccuracyEvaluator;
import net.jkernelmachines.evaluation.RandomSplitCrossValidation;
import net.jkernelmachines.kernel.typed.DoubleGaussL2;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.generators.GaussianGenerator;

import org.junit.Before;
import org.junit.Test;

/**
 * @author picard
 *
 */
public class RandomSplitCrossValidationTest {


	List<TrainingSample<double[]>> train;
	LaSVM<double[]> svm;

	@Before
	public void setUp() throws Exception {
		
		GaussianGenerator g = new GaussianGenerator(10, 5.0f, 0.1);
		train = g.generateList(50);
		
		DoubleGaussL2 k = new DoubleGaussL2(1.0);
		svm = new LaSVM<double[]>(k);
		svm.setC(1.0);
	}


	/**
	 * Test method for {@link net.jkernelmachines.evaluation.RandomSplitCrossValidation#run()}.
	 */
	@Test
	public final void testRun() {
		AccuracyEvaluator<double[]> ae = new AccuracyEvaluator<double[]>();
		RandomSplitCrossValidation<double[]> rscv = new RandomSplitCrossValidation<double[]>(svm, train, ae);
		rscv.setTrainPercent(0.9);
		rscv.setNbTest(10);

		rscv.run();
		assertEquals(1.0, rscv.getAverageScore(), 1e-15);

		rscv.setBalanced(false);
		rscv.run();
		assertEquals(1.0, rscv.getAverageScore(), 1e-15);
	}

}
