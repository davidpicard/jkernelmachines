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

    Copyright David Picard - 2012

 */
package fr.lip6.evaluation;

import java.util.List;

import fr.lip6.classifier.Classifier;
import fr.lip6.classifier.multiclass.MulticlassClassifier;
import fr.lip6.type.TrainingSample;
import fr.lip6.util.DebugPrinter;

/**
 * Evaluation class for computing the multiclass accuracy on a testing set,
 * given a provided cmulticlass classifier.
 * 
 * @author picard
 * 
 */
public class MulticlassAccuracyEvaluator<T> implements Evaluator<T> {

	MulticlassClassifier<T> classifier;
	List<TrainingSample<T>> trainList;
	List<TrainingSample<T>> testList;
	double accuracy;

	DebugPrinter debug = new DebugPrinter();

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.evaluation.Evaluator#setClassifier(fr.lip6.classifier.Classifier)
	 */
	@Override
	public void setClassifier(Classifier<T> cls) {
		if (cls instanceof MulticlassClassifier<?>) {
			classifier = (MulticlassClassifier<T>) cls;
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.evaluation.Evaluator#setTrainingSet(java.util.List)
	 */
	@Override
	public void setTrainingSet(List<TrainingSample<T>> trainlist) {
		this.trainList = trainlist;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.evaluation.Evaluator#setTestingSet(java.util.List)
	 */
	@Override
	public void setTestingSet(List<TrainingSample<T>> testlist) {
		this.testList = testlist;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.evaluation.Evaluator#evaluate()
	 */
	@Override
	public void evaluate() {
		if (classifier == null || trainList == null || testList == null) {
			accuracy = -1;
			return;
		}
		if (trainList.isEmpty() || testList.isEmpty()) {
			accuracy = -1;
			return;
		}

		// train
		classifier.train(trainList);

		double top = 0;
		for (TrainingSample<T> t : testList) {
			if (classifier.valueOf(t.sample) == t.label)
				top++;
		}
		accuracy = top / testList.size();
		debug.println(2, "Multiclass accuracy: "+accuracy);

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.evaluation.Evaluator#getScore()
	 */
	@Override
	public double getScore() {
		return accuracy;
	}

}
