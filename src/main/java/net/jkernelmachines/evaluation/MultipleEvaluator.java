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
