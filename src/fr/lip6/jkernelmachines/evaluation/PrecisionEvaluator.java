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

    Copyright David Picard - 2013

*/
package fr.lip6.jkernelmachines.evaluation;

import java.util.List;

import fr.lip6.jkernelmachines.classifier.Classifier;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * @author picard
 *
 */
public class PrecisionEvaluator<T> implements Evaluator<T> {
	
	private Classifier<T> cls;
	private List<TrainingSample<T>> trainList;
	private List<TrainingSample<T>> testList;
	
	

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
		this.trainList = trainlist;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.evaluation.Evaluator#setTestingSet(java.util.List)
	 */
	@Override
	public void setTestingSet(List<TrainingSample<T>> testlist) {
		this.testList = testlist;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.evaluation.Evaluator#evaluate()
	 */
	@Override
	public void evaluate() {
		cls.train(trainList);
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.evaluation.Evaluator#getScore()
	 */
	@Override
	public double getScore() {

		double prec;
		int tp = 0, fp = 0;
		
		for(TrainingSample<T> t : testList) {
			double v = cls.valueOf(t.sample);
			
			if(v >= 0) {
				if(t.label >= 0) {
					tp++;
				}
				else {
					fp++;
				}
			}			
		}
		prec = 0;
		if(tp+fp > 0) {
			prec = tp/((double)tp+fp);
		}
		
		return  prec;
	}

}
