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
package net.jkernelmachines.classifier.multiclass;

import java.util.Map;

import net.jkernelmachines.classifier.Classifier;

/**
 * Interface for multiclass classifiers.
 * 
 * @author picard
 *
 */
public interface MulticlassClassifier<T> extends Classifier<T> {
	
	/**
	 * Tells the confidence associated with the predicted class
	 * @param t the sample to evaluate
	 * @return the confidence
	 */
	public double getConfidence(T t);
	
	/**
	 * Tells the confidences for all classes
	 * @param t the sample to evaluate
	 * @return the vector of confidences for all classes
	 */
	public Map<Integer, Double> getConfidences(T t);

}
