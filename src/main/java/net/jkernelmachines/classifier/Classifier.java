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
package net.jkernelmachines.classifier;

import java.util.List;

import net.jkernelmachines.type.TrainingSample;

/**
 * Classifier interface that provides training and evaluation methods.
 * @author dpicard
 *
 * @param <T>
 */
public interface Classifier<T> extends Cloneable {
	
	/**
	 * Replace the current training list and train the classifier
	 * @param l list of training samples
	 */
	public void train(List<TrainingSample<T>> l);
	
	/**
	 * Computes the category of the provided example
	 * @param e example
	 * @return &gt; 0. if e belongs to the category,  &lt; 0. if not.
	 */
	public double valueOf(T e);
	
	/**
	 * Creates and returns a copy of this object.
	 * @see java.lang.Object#clone()
	 */
	public Classifier<T> copy() throws CloneNotSupportedException;
}
