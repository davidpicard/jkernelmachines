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
package net.jkernelmachines.kernel.adaptative;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import net.jkernelmachines.kernel.Kernel;
import net.jkernelmachines.threading.ThreadedMatrixOperator;
import net.jkernelmachines.type.TrainingSample;

/**
 * Major kernel computed as a weighted product of minor kernels : 
 * K = k_i^{w_i}
 * Computation of the kernel matrix is done by running a thread on sub matrices.
 * The number of threads is chosen as function of the number of available cpus.
 * @author dpicard
 *
 * @param <T> samples datatype
 */
public class ThreadedProductKernel<T> extends Kernel<T> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7780445301175174296L;
	
	
	private Map<Kernel<T>, Double> kernels;
	protected int numThread = 0;
	
	public ThreadedProductKernel()
	{
		kernels = new HashMap<Kernel<T>, Double>();
	}

	/**
	 * Sets the weights to h.
	 * @param h weights map
	 */
	public ThreadedProductKernel(Map<Kernel<T>, Double> h)
	{
		kernels = new HashMap<Kernel<T>, Double>();
		kernels.putAll(h);
	}
	
	/**
	 * adds a kernel to the sum with weight 1.0
	 * @param k kernel
	 */
	public void addKernel(Kernel<T> k)
	{
		synchronized(kernels)
		{
			kernels.put(k, 1.0);
		}
	}
	
	/**
	 * adds a kernel to the sum with weight d
	 * @param k kernel
	 * @param d weight
	 */
	public void addKernel(Kernel<T> k , double d)
	{
		synchronized(kernels)
		{
			kernels.put(k, d);
		}
	}
	
	/**
	 * removes kernel k from the sum
	 * @param k kernel
	 */
	public void removeKernel(Kernel<T> k)
	{
		synchronized(kernels)
		{
			kernels.remove(k);
		}
	}
	
	/**
	 * gets the weights of kernel k
	 * @param k kernel
	 * @return the weight associated with k
	 */
	public double getWeight(Kernel<T> k)
	{
		synchronized(kernels)
		{
			Double d = kernels.get(k);
			if(d == null)
				return 0.;
			return d.doubleValue();
		}
	}
	
	/**
	 * Sets the weight of kernel k
	 * @param k kernel
	 * @param d weight
	 */
	public void setWeight(Kernel<T> k, Double d)
	{
		synchronized(kernels)
		{
			kernels.put(k, d);
		}
	}
	
	@Override
	public double valueOf(T t1, T t2) {
		double sum = 1.;
		for(Kernel<T> k : kernels.keySet())
		{
			double w = kernels.get(k);
			if(w != 0)
				sum *= Math.pow(k.valueOf(t1, t2), kernels.get(k));
		}
		
		return sum;
	}

	@Override
	public double valueOf(T t1) {
		return valueOf(t1, t1);
	}
	
	/**
	 * get the list of kernels and associated weights.
	 * @return hashtable containing kernels as keys and weights as values.
	 */
	public Map<Kernel<T>, Double> getWeights()
	{
		return kernels;
	}
	
	@Override
	public double[][] getKernelMatrix(List<TrainingSample<T>> list)
	{
		final List<TrainingSample<T>> l = list;
		//init matrix with ones
		double matrix[][] = new double[l.size()][l.size()];
		for(double[] lines : matrix)
			Arrays.fill(lines, 1.);
		

		for(final Kernel<T> k : kernels.keySet())
		{
			final double w = kernels.get(k);
			
			//check w
			if(w == 0)
				continue;
			

			final double[][] m = k.getKernelMatrix(l);
			// specific factory
			ThreadedMatrixOperator tmo = new ThreadedMatrixOperator(){
				
				@Override
				public void doLines(double[][] matrix , int from , int to) {
					
					for(int index = from ; index < to ; index++)
					{
						for(int i = 0 ; i < m[index].length ; i++)
						{
							matrix[index][i] *= Math.pow(m[index][i], w);
						}
					}
				};
				
			};
			
			matrix = tmo.getMatrix(matrix);
		}
		
		return matrix;
	}
}
