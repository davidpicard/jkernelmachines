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
package fr.lip6.kernel;

import java.util.List;

import fr.lip6.threading.ThreadedMatrixOperator;
import fr.lip6.threading.ThreadPoolServer;
import fr.lip6.type.TrainingSample;

/**
 * <p>
 * Base class for Gaussian Kernels in the form of k(x1, x2) = exp(-gamme * dist(x1, x2))
 * </p>
 * <p>
 * The distance used if defined in specific subclasses.
 * </p>
 * @author picard
 *
 * @param <T> Data type of input space
 */
public abstract class GaussianKernel<T> extends Kernel<T> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -4741333152317305622L;

	/**
	 * Sets exponential coefficient. 
	 *
	 * @param gamma gamma coefficient
	 */
	public abstract void setGamma(double gamma);
	
	/**
	 * Tells exponential coefficient
	 * @return gamma
	 */
	public abstract double getGamma();
	
	/**
	 * Tells the inner distance between two samples used by this Gaussian kernel.
	 * @param t1 first sample
	 * @param t2 second sample
	 * @return the distance between the two samples
	 */
	public abstract double distanceValueOf(T t1, T t2);
	
	/**
	 * Tells the distance matrix for a specified list of samples.<br />
	 * This is a threaded operation.
	 * @param l the list of samples
	 * @return the distance matrix
	 */
	public double[][] getDistanceMatrix(final List<TrainingSample<T>> l)
	{
		double[][] matrix = new double[l.size()][l.size()];
		
		//computing matrix				
		ThreadedMatrixOperator factory = new ThreadedMatrixOperator()
		{
			@Override
			public void doLines(double[][] matrix, int from, int to) {
				for(int index = from ; index < to ; index++)
				{
					T s1 = l.get(index).sample;
					for(int j = 0 ; j < matrix.length ; j++)
						matrix[index][j] = distanceValueOf(s1, l.get(j).sample);
				}
			}
		};
		
		factory.getMatrix(matrix);
		
		ThreadPoolServer.shutdownNow();	
		return matrix;
	}
}
