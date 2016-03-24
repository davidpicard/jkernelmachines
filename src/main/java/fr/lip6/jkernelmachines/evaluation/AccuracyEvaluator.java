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
package fr.lip6.jkernelmachines.evaluation;

import java.util.List;

import fr.lip6.jkernelmachines.classifier.Classifier;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.DebugPrinter;

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
