/**
    This file is part of JkernelMachines.

    JkernelMachines is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    JkernelMachines is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with JkernelMachines.  If not, see <http://www.gnu.org/licenses/>.

    Copyright David Picard - 2014

*/
package fr.lip6.jkernelmachines.active;

import static java.lang.Math.abs;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import fr.lip6.jkernelmachines.classifier.OnlineClassifier;
import fr.lip6.jkernelmachines.classifier.multiclass.MulticlassClassifier;
import fr.lip6.jkernelmachines.type.TrainingSample;

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
