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

/**
 * Basic interface for all evaluation tools.
 * @author picard
 *
 * @param <T> samples data type
 */
public interface Evaluator<T> {

	/**
	 * Sets the classifier to use for evaluation
	 * @param cls the classifier
	 */
	public void setClassifier(Classifier<T> cls);
	
	/**
	 * Sets the list of training samples on which to train the classifier
	 * @param trainlist the training set
	 */
	public void setTrainingSet(List<TrainingSample<T>> trainlist);
	
	/**
	 * Sets the list of testing samples on which to evaluate the classifier
	 * @param testlist the testing set
	 */
	public void setTestingSet(List<TrainingSample<T>> testlist);
	
	/**
	 * Run the training procedure and compute score.
	 */
	public void evaluate();
	
	/**
	 * Tells the score resulting of the evaluation
	 * @return the score
	 */
	public double getScore();
	
}
