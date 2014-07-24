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

import java.util.List;
import java.util.Map;

import fr.lip6.jkernelmachines.kernel.Kernel;

/**
 * Interface for Multiple Kernel Classes. Provides useful methods for
 * retrieveing multiple kernels related information.
 * 
 * @author picard
 * 
 */
public interface MKL<T> {
	
	/**
	 * Adds a kernel to the MKL problem
	 * @param kernel the new kernel to add
	 */
	public void addKernel(Kernel<T> kernel);

	/**
	 * Gets an array containing the weights of the different kernels, in the
	 * same order as getKernels()
	 * 
	 * @return the array of weights
	 */
	public double[] getKernelWeights();

	/**
	 * Gets an array of the kernels in the set, in the same order as
	 * getKernelWeights()
	 * 
	 * @return the array of kernels
	 */
	public List<Kernel<T>> getKernels();

	/**
	 * Gets a mapping of pairs <Kernel,Double> containing the kernels and
	 * weights in the set.
	 * 
	 * @return the map of pairs <kernel,weight>
	 */
	public Map<Kernel<T>, Double> getKernelWeightMap();

}
