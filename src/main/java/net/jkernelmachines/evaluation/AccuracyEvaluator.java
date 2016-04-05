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
package net.jkernelmachines.evaluation;

import java.util.List;

import net.jkernelmachines.classifier.Classifier;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.DebugPrinter;

/**
 * Simple evaluation class for computing the accuracy on a testing set.
 *
 * @param <T> the type of data samples
 */
public class AccuracyEvaluator<T> implements Evaluator<T> {

	Classifier<T> classifier;
	List<TrainingSample<T>> trainList;
	List<TrainingSample<T>> testList;
	double accuracy;
	
	DebugPrinter debug = new DebugPrinter();
	
	/* (non-Javadoc)
	 * @see fr.lip6.evaluation.Evaluator#setClassifier(fr.lip6.classifier.Classifier)
	 */
	@Override
	public void setClassifier(Classifier<T> cls) {
		classifier = cls;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.evaluation.Evaluator#setTrainingSet(java.util.List)
	 */
	@Override
	public void setTrainingSet(List<TrainingSample<T>> trainlist) {
		this.trainList = trainlist;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.evaluation.Evaluator#setTestingSet(java.util.List)
	 */
	@Override
	public void setTestingSet(List<TrainingSample<T>> testlist) {
		this.testList = testlist;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.evaluation.Evaluator#evaluate()
	 */
	@Override
	public void evaluate() {
		if(trainList != null) {
			long time = System.currentTimeMillis();
			classifier.train(trainList);
			debug.println(2, "trained in "+(System.currentTimeMillis()-time)+"ms.");
		}
		if(testList != null) {
			long time = System.currentTimeMillis();
			double good = 0;
			for(TrainingSample<T> t : testList) {
				double v = classifier.valueOf(t.sample);
				if(v*t.label > 0)
					good++;
			}
			accuracy = good / (double) testList.size();
			debug.println(2, "evaluation done in "+(System.currentTimeMillis()-time)+"ms.");
		}
	}

	/* (non-Javadoc)
	 * @see fr.lip6.evaluation.Evaluator#getScore()
	 */
	@Override
	public double getScore() {
		return accuracy;
	}

}
