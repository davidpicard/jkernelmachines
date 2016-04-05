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

import static java.lang.Math.abs;

import java.util.ArrayList;
import java.util.List;

import net.jkernelmachines.classifier.OnlineClassifier;
import net.jkernelmachines.type.TrainingSample;

/**
 * <p>
 * Simple active learning strategy as presented in:
 * 
 * Support vector machine active learning with applications to text classification. 
 * S. Tong and D. Koller.
 * Journal of Machine Learning Research, 2:45–66, 2001.
 * </p>
 * @author picard
 *
 */
public class SimpleAL<T> extends ActiveLearner<T> {
	
	public SimpleAL(OnlineClassifier<T> c, List<TrainingSample<T>> l) {
		classifier = c;
		train = new ArrayList<TrainingSample<T>>(l.size());
		train.addAll(l);
	}
	
	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.active.ActiveLearner#getActiveSample(java.util.List)
	 */
	@Override
	public TrainingSample<T> getActiveSample(List<TrainingSample<T>> l) {
		if(classifier == null) {
			return null;
		}
		
		double min = Double.POSITIVE_INFINITY;
		int index = -1;
		
		for(int i = 0 ; i < l.size() ; i++) {
			double v = abs(classifier.valueOf(l.get(i).sample));
			if(v < min) {
				min = v;
				index = i;
			}
		}
		
		return l.get(index);
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.active.ActiveLearner#updateClassifier(int)
	 */
	@Override
	public void updateClassifier(int nbSamples) {
		if(classifier == null) {
			return;
		}
		
		for(int i = 0 ; i < nbSamples ; i++) {
			if(train.isEmpty()) {
				return;
			}
			
			TrainingSample<T> t = getActiveSample(train);
			if(t == null) {
				return;
			}
			train.remove(t);
			classifier.train(t);
		}
		
	}

}
