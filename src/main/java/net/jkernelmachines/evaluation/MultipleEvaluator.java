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
package net.jkernelmachines.evaluation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import net.jkernelmachines.classifier.Classifier;
import net.jkernelmachines.type.TrainingSample;

/**
 * Class that aggregates several Evaluators
 * @author picard
 *
 */
public class MultipleEvaluator<T> implements Evaluator<T> {
	
	Classifier<T> cls;
	Map<String, Evaluator<T>> evaluators = new HashMap<>();
	List<TrainingSample<T>> trainList;
	
	/**
	 * Adds an evaluator to the list of things being computed
	 * @param name string associated with the evaluator
	 * @param e the evaluator
	 */
	public void addEvaluator(String name, Evaluator<T> e) {
		if(!evaluators.containsKey(name)) {
			evaluators.put(name, e);
		}
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.evaluation.Evaluator#setClassifier(fr.lip6.jkernelmachines.classifier.Classifier)
	 */
	@Override
	public void setClassifier(Classifier<T> cls) {
		this.cls = cls;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.evaluation.Evaluator#setTrainingSet(java.util.List)
	 */
	@Override
	public void setTrainingSet(List<TrainingSample<T>> trainlist) {
		this.trainList = new ArrayList<TrainingSample<T>>(trainList.size());
		this.trainList.addAll(trainList);
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.evaluation.Evaluator#setTestingSet(java.util.List)
	 */
	@Override
	public void setTestingSet(List<TrainingSample<T>> testlist) {
		for(Evaluator<T> e : evaluators.values()) {
			e.setTestingSet(testlist);
		}
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.evaluation.Evaluator#evaluate()
	 */
	@Override
	public void evaluate() {
		cls.train(trainList);
		for(Evaluator<T> e : evaluators.values()) {
			e.setClassifier(cls);
			e.evaluate();
		}
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.evaluation.Evaluator#getScore()
	 */
	@Override
	public double getScore() {
		return 0; //useless
	}
	
	public double getScore(String name) {
		if(evaluators.containsKey(name)) {
			return evaluators.get(name).getScore();
		}
		return 0;
	}

}
