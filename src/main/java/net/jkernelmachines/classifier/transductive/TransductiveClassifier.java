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

    Copyright David Picard - 2010

*/
package net.jkernelmachines.classifier.transductive;

import java.util.List;

import net.jkernelmachines.type.TrainingSample;

/**
 * Interface for transductive classifiers.
 * @author dpicard
 *
 * @param <T> Datatype of input space
 */
public interface TransductiveClassifier<T> {
	
	/**
	 * Train the classifier on trainList, with the help of testList in a transductive way.
	 * @param trainList
	 * @param testList
	 */
	public void train(List<TrainingSample<T>> trainList, List<TrainingSample<T>> testList);
	
	/**
	 * prediction output for t.
	 * @param t sample to evaluate
	 * @return the output value for this sample
	 */
	public double valueOf(T t);

}
