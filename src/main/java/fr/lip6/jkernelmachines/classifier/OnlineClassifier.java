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
package fr.lip6.jkernelmachines.classifier;

import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.type.TrainingSampleStream;

/**
 * Interface for classifier that are trainable from a stream of samples
 * 
 * @author picard
 *
 */
public interface OnlineClassifier<T> extends Classifier<T> {

	/**
	 * Add a single example to the current training set and train the classifier
	 * 
	 * @param t
	 *            the training sample
	 */
	public void train(TrainingSample<T> t);

	/**
	 * Train the classifier using a stream of TrainingSample sampled from the
	 * TrainingSampleStream until no sample can be drawn.
	 * 
	 * @param stream
	 *            the TrainingSampleStream from which the samples are drawn.
	 */
	public void onlineTrain(TrainingSampleStream<T> stream);

}
