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
package net.jkernelmachines.kernel.extra.bag;

import java.io.Serializable;
import java.util.List;

import net.jkernelmachines.kernel.Kernel;

/**
 * Kernel on bags of same length. 
 * Let B1 and B2 be bags of elements b1[i] and b2[i], let k(b1[i],b2[i]) be the minor kernel, then K(B1, B2) = sum_{i=n}^{N} k(b1[i],b2[i])
 * With n and N being the bounds of the sum.
 * @author dpicard
 *
 * @param <S>
 * @param <T>
 */
public class SimpleSubListKernel<S,T extends List<S>> extends Kernel<T> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1591055803491554966L;
	
	private int from = 0;
	private int to = 0;
	private Kernel<S> kernel;
	
	

	

	/**
	 * @param from starting point of the sum
	 * @param to end poitn of the sum
	 * @param kernel minor kernel
	 */
	public SimpleSubListKernel(int from, int to, Kernel<S> kernel) {
		this.from = from;
		this.to = to;
		this.kernel = kernel;
	}

	@Override
	public double valueOf(T t1, T t2) {
		double sum = 0;
		if(to > t1.size())
			to = t1.size();
		if(to > t2.size())
			to = t2.size();
		
		for(int i = from; i < to; i++)
			sum += kernel.valueOf(t1.get(i), t2.get(i));
		
		return sum/((double)(to-from));
	}

	@Override
	public double valueOf(T t1) {
		return valueOf(t1, t1);
	}



	
	
}

	

