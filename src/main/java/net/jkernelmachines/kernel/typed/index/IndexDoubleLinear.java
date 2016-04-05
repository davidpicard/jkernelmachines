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
package net.jkernelmachines.kernel.typed.index;

import net.jkernelmachines.kernel.Kernel;

/**
 * Kernel on double[] that performs the product of a specified component j:
 * k(x,y) = x[j]*y[j]
 * @author dpicard
 *
 */
public class IndexDoubleLinear extends Kernel<double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 457509780005148420L;
	
	
	private int ind = 0;
	
	/**
	 * Constructor specifying the component which is used
	 * @param feature the index of the component
	 */
	public IndexDoubleLinear(int feature)
	{
		ind = feature;
	}
	
	@Override
	public double valueOf(double[] t1, double[] t2) {
		if(t1[ind] == 0. || t2[ind] == 0.)
			return 0.;
		return t2[ind]*t1[ind];
	}

	@Override
	public double valueOf(double[] t1) {

		if(t1[ind] == 0.)
			return 0.;
		return t1[ind]*t1[ind];
	}

	
	public void setIndex(int i)
	{
		this.ind = i;
	}

}
