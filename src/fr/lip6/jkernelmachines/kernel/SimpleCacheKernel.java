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
package fr.lip6.jkernelmachines.kernel;

import java.util.List;

import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * Very simple caching method for any kernel. Caches only the Gram matrix of a specified list of training samples.
 * @author picard
 *
 * @param <T>
 */
public final class SimpleCacheKernel<T> extends Kernel<T> {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = -2417905029129394427L;
	
	final private Kernel<T> kernel;
	final private double matrix[][];
	
	/**
	 * Constructor using a kernel and a list of samples
	 * @param k the underlying of this caching kernel
	 * @param l the list on which to compute the Gram matrix
	 */
	public SimpleCacheKernel(Kernel<T> k, List<TrainingSample<T>> l) {
		kernel = k;
		matrix = k.getKernelMatrix(l);
	}
	
	
	@Override
	final public double valueOf(T t1, T t2) {
		return kernel.valueOf(t1, t2);
	}

	@Override
	final public double valueOf(T t1) {
		return kernel.valueOf(t1);
	}


	@Override
	public double[][] getKernelMatrix(List<TrainingSample<T>> e) {
		return matrix;
	}
	
	

}
