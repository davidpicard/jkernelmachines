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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import net.jkernelmachines.classifier.OnlineClassifier;
import net.jkernelmachines.type.TrainingSample;

/**
 * <p>
 * Absurd active strategy that randomly selects a sample
 * </p>
 * 
 * @author picard
 * 
 */
public class RandomAL<T> extends ActiveLearner<T> {

	Random rand;

	public RandomAL(OnlineClassifier<T> c, List<TrainingSample<T>> l) {
		classifier = c;
		train = new ArrayList<TrainingSample<T>>(l.size());
		train.addAll(l);
		rand = new Random();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.active.ActiveLearner#getActiveSample(java.util
	 * .List)
	 */
	@Override
	public TrainingSample<T> getActiveSample(List<TrainingSample<T>> l) {
		if (classifier == null) {
			return null;
		}

		int index = rand.nextInt(l.size());
		return l.get(index);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.active.ActiveLearner#updateClassifier(int)
	 */
	@Override
	public void updateClassifier(int nbSamples) {
		if (classifier == null) {
			return;
		}

		for (int i = 0; i < nbSamples; i++) {
			if (train.isEmpty()) {
				return;
			}

			TrainingSample<T> t = getActiveSample(train);
			if (t == null) {
				return;
			}
			train.remove(t);
			classifier.train(t);
		}

	}

}
