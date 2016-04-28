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
 * Simple multithreaded implementation over a given Kernel. The multithreading comes only when
 * computing the Gram matrix.
 * Number of Threads is function of available processors.
 * @author dpicard
 *
 * @param <T> samples datatype
 */
public class ThreadedKernel<T> extends Kernel<T> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2193768216118832033L;
	
	
	protected final Kernel<T> k;

	/**
	 * MultiThread the given kernel
	 * @param kernel underlying kernel
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

					for(int j = index ; j < matrix[index].length ; j++)
					{
						matrix[index][j] = k.valueOf(xi, l.get(j).sample);
						matrix[j][index] = matrix[index][j];
					}
				}
			};
		};

		/* do the actuel computing of the matrix */
		matrix = factory.getMatrix(matrix);
		
		return matrix;
	}



}
