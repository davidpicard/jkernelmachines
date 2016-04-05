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

/**
 * Basic interface for all evaluation tools.
 * @author picard
 *
 * @param <T> samples data type
 */
public interface Evaluator<T> {

	/**
	 * Sets the classifier to use for evaluation
	 * @param cls the classifier
	 */
	public void setClassifier(Classifier<T> cls);
	
	/**
	 * Sets the list of training samples on which to train the classifier
	 * @param trainlist the training set
	 */
	public void setTrainingSet(List<TrainingSample<T>> trainlist);
	
	/**
	 * Sets the list of testing samples on which to evaluate the classifier
	 * @param testlist the testing set
	 */
	public void setTestingSet(List<TrainingSample<T>> testlist);
	
	/**
	 * Run the training procedure and compute score.
	 */
	public void evaluate();
	
	/**
	 * Tells the score resulting of the evaluation
	 * @return the score
	 */
	public double getScore();
	
}
