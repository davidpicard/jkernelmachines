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
import net.jkernelmachines.classifier.multiclass.OneAgainstAll;
import net.jkernelmachines.evaluation.MulticlassAccuracyEvaluator;
import net.jkernelmachines.kernel.typed.DoubleGaussL2;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.generators.MultiClassGaussianGenerator;

import org.junit.Before;
import org.junit.Test;

/**
 * @author picard
 *
 */
public class MulticlassAccuracyEvaluatorTest {

	List<TrainingSample<double[]>> train;
	OneAgainstAll<double[]> multisvm;
	
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		MultiClassGaussianGenerator mcgg = new MultiClassGaussianGenerator(4);
		mcgg.setP(10);
		mcgg.setSigma(1);
		train = mcgg.generateList(5);
		
		DoubleGaussL2 k = new DoubleGaussL2();
		k.setGamma(0.5);
		LaSVM<double[]> svm = new LaSVM<double[]>(k);
		svm.setC(10);
		multisvm = new OneAgainstAll<double[]>(svm);
	}

	/**
	 * Test method for {@link net.jkernelmachines.evaluation.MulticlassAccuracyEvaluator#evaluate()}.
	 */
	@Test
	public final void testEvaluate() {
		MulticlassAccuracyEvaluator<double[]> ae = new MulticlassAccuracyEvaluator<double[]>();
		ae.setClassifier(multisvm);
		ae.setTrainingSet(train);
		ae.setTestingSet(train);
		
		ae.evaluate();
		
		assertEquals(1.0, ae.getScore(), 1e-15);	
	}

}
