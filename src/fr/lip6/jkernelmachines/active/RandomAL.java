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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import fr.lip6.jkernelmachines.classifier.Classifier;
import fr.lip6.jkernelmachines.type.TrainingSample;

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

	public RandomAL(Classifier<T> c, List<TrainingSample<T>> l) {
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
