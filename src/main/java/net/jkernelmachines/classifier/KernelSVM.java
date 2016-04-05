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
package net.jkernelmachines.classifier;

import net.jkernelmachines.kernel.Kernel;

/**
 * Interface for SVM algorithms using a non-linear kernel (dual optimization mainly)
 * 
 * @author picard
 *
 */
public interface KernelSVM<T> extends Classifier<T> {
    
        /**
         * Tells the current Kernel.
         * @return the current kernel
         */
        public Kernel<T> getKernel();
	
	/**
	 * Sets the kernel to use as similarity measure
	 * @param k the kernel function
	 */
	public void setKernel(Kernel<T> k);
	
	/**
	 * Tells the weights of training samples
	 * 
	 * @return an array of double representing the weights
	 */
	public double[] getAlphas();
	
	/**
	 * Sets the hyperparameter C
	 * @param c the hyperparameter C
	 */
	public void setC(double c);
	
	/**
	 * Tells the hyperparameter C
	 * @return the hyperparameter C
	 */
	public double getC();

}
