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
package net.jkernelmachines.density;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import net.jkernelmachines.kernel.Kernel;

/**
 * Parzen window for estimating the probability density function of a random variable.
 * @author dpicard
 *
 * @param <T> Datatype of input space
 */
public class ParzenDensity<T> implements DensityFunction<T>, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5414922333951533146L;
	
	
	private Kernel<T> kernel;
	ArrayList<T> set;
	
	public ParzenDensity(Kernel<T> kernel)
	{
		this.kernel = kernel;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.density.DensityFunction#train(java.lang.Object)
	 */
	public void train(T e) {
		if(set == null)
		{
			set = new ArrayList<T>();
		}

		set.add(e);
				
	}

	/* (non-Javadoc)
	 * @see fr.lip6.density.DensityFunction#train(T[])
	 */
	public void train(List<T> e) {
		if(set == null)
		{
			set = new ArrayList<T>();
		}
		
		for(T t : e)
			set.add(t);
	}

	/* (non-Javadoc)
	 * @see fr.lip6.density.DensityFunction#valueOf(java.lang.Object)
	 */
	public double valueOf(T e) {

		double sum = 0.;
		for(int i = 0 ; i < set.size(); i++)
			sum += kernel.valueOf(set.get(i), e);
		
		sum /= set.size();
		
		return sum;
	}

}
