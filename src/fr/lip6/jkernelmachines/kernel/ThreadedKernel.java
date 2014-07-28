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

import fr.lip6.jkernelmachines.threading.ThreadedMatrixOperator;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * Simple multithreaded implementation over a given Kernel. The multithreading comes only when
 * computing the Gram matrix.<br />
 * Number of Threads is function of available processors.
 * @author dpicard
 *
 * @param <T>
 */
public class ThreadedKernel<T> extends Kernel<T> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2193768216118832033L;
	
	
	protected final Kernel<T> k;

	/**
	 * MultiThread the given kernel
	 * @param kernel
	 */
	public ThreadedKernel(Kernel<T> kernel)
	{
		this.k = kernel;
	}
	

	@Override
	public double valueOf(T t1, T t2) {
		return k.valueOf(t1, t2);
	}

	@Override
	public double valueOf(T t1) {
		return k.valueOf(t1);
	}
	
	
	@Override
	public double[][] getKernelMatrix(final List<TrainingSample<T>> l) {
		
		final List<TrainingSample<T>> e = l;
		double[][] matrix = new double[e.size()][e.size()];
				
		ThreadedMatrixOperator factory = new ThreadedMatrixOperator()
		{
			@Override
			public void doLines(double[][] matrix, int from, int to) {
				for(int index = from ; index < to ; index++)
				{
					T xi = l.get(index).sample;

					for(int j = 0 ; j < matrix[index].length ; j++)
					{
						matrix[index][j] = k.valueOf(xi, l.get(j).sample);
					}
				}
			};
		};

		/* do the actuel computing of the matrix */
		matrix = factory.getMatrix(matrix);
		
		return matrix;
	}



}
