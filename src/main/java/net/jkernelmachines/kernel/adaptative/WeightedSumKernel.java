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

import java.util.Hashtable;
import java.util.List;

import net.jkernelmachines.kernel.Kernel;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.algebra.VectorOperations;

/**
 * Major kernel computed as a weighted sum of minor kernels : 
 * K = w_i * k_i
 * Non-threaded version
 * @author dpicard
 *
 * @param <T> input space datatype
 */
public class WeightedSumKernel<T> extends Kernel<T> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4590492743843223113L;
	
	
	private Hashtable<Kernel<T>, Double> kernels;
	
	public WeightedSumKernel()
	{
		kernels = new Hashtable<Kernel<T>, Double>();
	}
	
	/**
	 * Sets the weights to h. Beware! It does not make a copy of h!
	 * @param h weights map
	 */
	public WeightedSumKernel(Hashtable<Kernel<T>, Double> h)
	{
		kernels = h;
	}
	
	/**
	 * adds a kernel to the sum with weight 1.0
	 * @param k kernel
	 */
	public void addKernel(Kernel<T> k)
	{
		kernels.put(k, 1.0);
	}
	
	/**
	 * adds a kernel to the sum with weight d
	 * @param k kernel
	 * @param d weight
	 */
	public void addKernel(Kernel<T> k , double d)
	{
		kernels.put(k, d);
	}
	
	/**
	 * removes kernel k from the sum
	 * @param k kernel
	 */
	public void removeKernel(Kernel<T> k)
	{
		kernels.remove(k);
	}
	
	/**
	 * gets the weights of kernel k
	 * @param k kernel
	 * @return the weight associated with k
	 */
	public double getWeight(Kernel<T> k)
	{
		Double d = kernels.get(k);
		if(d == null)
			return 0.;
		return d.doubleValue();
	}
	
	/**
	 * Sets the weight of kernel k
	 * @param k kernel
	 * @param d weight
	 */
	public void setWeight(Kernel<T> k, Double d)
	{
		kernels.put(k, d);
	}
	
	@Override
	public double valueOf(T t1, T t2) {
		double sum = 0.;
		for(Kernel<T> k : kernels.keySet())
			sum += kernels.get(k)*k.valueOf(t1, t2);
		
		return sum;
	}

	@Override
	public double valueOf(T t1) {
		double sum = 0.;
		for(Kernel<T> k : kernels.keySet())
			sum += kernels.get(k)*k.valueOf(t1);
		
		return sum;
	}
	
	/**
	 * get the list of kernels and associated weights.
	 * @return hashtable containing kernels as keys and weights as values.
	 */
	public Hashtable<Kernel<T>, Double> getWeights()
	{
		return kernels;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.kernel.Kernel#getKernelMatrix(java.util.ArrayList)
	 */
	@Override
	public double[][] getKernelMatrix(List<TrainingSample<T>> e) {
		double matrix[][] = new double[e.size()][e.size()];
		
		for(Kernel<T> k : kernels.keySet())
		{
			double[][] m = k.getKernelMatrix(e);
			double w = kernels.get(k)/100;
			w = w*100;
			for(int i = 0 ; i < e.size() ; i++)
			for(int j = i ; j < e.size() ; j++)
			{
				matrix[i][j] += w*m[i][j];
				if(i != j)
					matrix[j][i] += w*m[j][i];
			}
		}
		
		return matrix;
	}

	@Override
	public double[] getKernelMatrixLine(T x, List<TrainingSample<T>> l) {
		double[] line = new double[l.size()];
		for(Kernel<T> k : kernels.keySet()) {
			VectorOperations.addi(line, line, kernels.get(k), k.getKernelMatrixLine(x, l));
		}
		return line;
	}
	
	

}
