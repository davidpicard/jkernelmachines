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
import java.util.Map;

import net.jkernelmachines.classifier.OnlineClassifier;
import net.jkernelmachines.classifier.multiclass.MulticlassClassifier;
import net.jkernelmachines.type.TrainingSample;

/**
 * <p>
 * Extension to multiclass of the Simple active learning strategy as presented in:
 * 
 * Support vector machine active learning with applications to text classification. 
 * S. Tong and D. Koller.
 * Journal of Machine Learning Research, 2:45–66, 2001.
 * </p>
 * @author picard
 *
 */
public class MulticlassSimpleAL<T> extends ActiveLearner<T> {
	
	List<TrainingSample<T>> used;
	int[] samplesPerClass;
	List<Integer> classes;
	boolean classeBalanced = true;
	
	@SuppressWarnings("unchecked")
	public MulticlassSimpleAL(MulticlassClassifier<T> c, List<TrainingSample<T>> l) {
		if(c instanceof OnlineClassifier<?>) {
			classifier = (OnlineClassifier<T>)c;
		}
		
		train = new ArrayList<TrainingSample<T>>(l.size());
		train.addAll(l);
		used = new ArrayList<TrainingSample<T>>();
		
		classes = new ArrayList<>();
		for(TrainingSample<T> t: train) {
			if(!classes.contains(t.label)) {
				classes.add(t.label);
			}
		}
		samplesPerClass = new int[classes.size()];
	}
	
	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.active.ActiveLearner#getActiveSample(java.util.List)
	 */
	@SuppressWarnings("unchecked")
	@Override
	public TrainingSample<T> getActiveSample(List<TrainingSample<T>> l) {
		if(classifier == null) {
			return null;
		}
		
		// find most imbalanced class
		int smin = 0;			
		if(classeBalanced) {
			for(int i = 1 ; i < samplesPerClass.length ; i++) {
				if(samplesPerClass[i] < samplesPerClass[smin]) {
					smin = i;
				}
			}
		}
		double min = Double.POSITIVE_INFINITY;
		int index = 0;
		
		for(int i = 0 ; i < l.size() ; i++) {
			TrainingSample<T> t = l.get(i);
			
			if(classeBalanced) {
				if(classes.indexOf(t.label) != smin)
					continue;
			}
			

			double v = ((MulticlassClassifier<T>)classifier).getConfidence(t.sample);
			double c = classifier.valueOf(t.sample);
			Map<Integer, Double> values = ((MulticlassClassifier<T>)classifier).getConfidences(t.sample);
			if(values == null) {
				continue;
			}
			for(int y : values.keySet()) {
				if(y != c) {
					double diff = abs(values.get(y) - v);
					if(diff < min) {
						min = diff;
						index = i;
					}
				}
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
			samplesPerClass[classes.indexOf(t.label)]++;
			
			classifier.train(t);
		}
		
	}

	@Override
	public void setClassifier(OnlineClassifier<T> cls) {
		if(!(cls instanceof MulticlassClassifier)) {
			throw new UnsupportedOperationException("Argument must be a MulticlassClassifier");
		}
		classifier = cls;
	}

	/**
	 * Tells is use a class balancing criterion
	 * @return true if criterion is used
	 */
	public boolean isClasseBalanced() {
		return classeBalanced;
	}

	/**
	 * Sets the use of a criterion aiming at balanced classes
	 * @param classeBalanced true if using the criterion
	 */
	public void setClasseBalanced(boolean classeBalanced) {
		this.classeBalanced = classeBalanced;
	}

}
