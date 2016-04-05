/*******************************************************************************
 * Copyright (c) 2016, David Picard.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************/
package net.jkernelmachines.kernel;

import java.util.List;

import net.jkernelmachines.threading.ThreadedMatrixOperator;
import net.jkernelmachines.type.TrainingSample;

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
	 * Tells the distance matrix for a specified list of samples.
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
		
		return matrix;
	}
}
