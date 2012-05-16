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
package fr.lip6.density;

import java.util.List;


/**
 * Density estimation based on training.
 * @author dpicard
 *
 */
public interface DensityFunction<T> {

	/**
	 * Adds a sample to the training set and train the density function
	 * @param e the sample to add to the training set
	 */
	public void train(T e);
	
	/**
	 * Train the density function on the specified training set
	 * @param e the list of training samples
	 */
	public void train(List<T> e);
	
	/**
	 * Value of the density function for the specified sample
	 * @param e the sample to evaluate
	 * @return value of the density function
	 */
	public double valueOf(T e);
	
}
