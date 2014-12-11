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

import fr.lip6.jkernelmachines.classifier.Classifier;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * <p>
 * Naïve active learning strategy that selects the most positive sample using
 * current classifier
 * </p>
 * 
 * @author picard
 * 
 */
public class BestAL<T> extends ActiveLearner<T> {

	public BestAL(Classifier<T> c, List<TrainingSample<T>> l) {
		classifier = c;
		train = new ArrayList<TrainingSample<T>>(l.size());
		train.addAll(l);
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

		double max = Double.NEGATIVE_INFINITY;
		int index = -1;

		for (int i = 0; i < l.size(); i++) {
			double v = abs(classifier.valueOf(l.get(i).sample));
			if (v > max) {
				max = v;
				index = i;
			}
		}

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
