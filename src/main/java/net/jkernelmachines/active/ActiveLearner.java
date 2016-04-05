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
package net.jkernelmachines.active;

import java.util.List;

import net.jkernelmachines.classifier.OnlineClassifier;
import net.jkernelmachines.type.TrainingSample;

/**
 * <p>
 * Base abstract class for active learning strategies. 
 * </p>
 * <p>
 * This class contains an instance
 * of a classifier and a training set of samples. The classifier can be updated externaly, 
 * or thanks to helpers present in the class 
 * </p>
 * @author picard
 *
 */
public abstract class ActiveLearner<T> {
		
	protected OnlineClassifier<T> classifier;
	protected List<TrainingSample<T>> train;
	
	/**
	 * Method returning the best training sample in the list with respect to the active strategy
	 * @return the best sample
	 */
	public abstract TrainingSample<T> getActiveSample(List<TrainingSample<T>> l);
	
	/**
	 * perform nbSample updates of the classifier using the active strategy
	 * @param nbSamples
	 */
	public abstract void updateClassifier(int nbSamples);
	
	/**
	 * Setter for the classifier
	 * @param cls
	 */
	public void setClassifier(OnlineClassifier<T> cls) {
		this.classifier = cls;
	}
	
	/**
	 * Getter for the classifier
	 * @return the classifier
	 */
	public OnlineClassifier<T> getClassifier() {
		return this.classifier;
	}

	/**
	 * Return the list of training samples
	 * @return the list of training samples
	 */
	public List<TrainingSample<T>> getTrain() {
		return train;
	}

	/**
	 * Sets the list of training samples
	 * @param train the list of training samples
	 */
	public void setTrain(List<TrainingSample<T>> train) {
		this.train = train;
	}

}
