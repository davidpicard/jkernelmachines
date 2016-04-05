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

import java.util.ArrayList;

import net.jkernelmachines.kernel.Kernel;

/**
 * performs a weighted product of several minor kernels, non threaded version.
 * @see ThreadedProductKernel
 * 
 * @author dpicard
 *
 * @param <T> data type of input space.
 */
public class WeightedProductKernel<T> extends Kernel<T> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6273022923321895693L;
	
	ArrayList<Kernel<T>> kernels;
	ArrayList<Double> weights;
	
	public WeightedProductKernel ()
	{
		super();
		
		kernels = new ArrayList<Kernel<T>>();
		weights = new ArrayList<Double>();
	}
	
	@Override
	public double valueOf(T t1, T t2) {
		
		double prod = 1.0;
		
		for(int i = 0 ; i < kernels.size(); i++)
			prod *= Math.pow(kernels.get(i).valueOf(t1, t2), weights.get(i));
		
		return prod;
	}

	@Override
	public double valueOf(T t1) {
		return valueOf(t1, t1);
	}
	
	/**
	 * adds a kernel to to product
	 * @param k
	 */
	public void addKernel(Kernel<T> k)
	{
		kernels.add(k);
		weights.add(1.0);
	}
	
	public void addKernel(Kernel<T> k, double w)
	{
		kernels.add(k);
		weights.add(w);
	}
	
	/**
	 * removes a kernel from the product
	 * @param k
	 */
	public void removeKernel(Kernel<T> k)
	{
		kernels.remove(k);
	}

}
