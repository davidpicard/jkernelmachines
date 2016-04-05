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
package net.jkernelmachines.kernel.typed;

import net.jkernelmachines.kernel.Kernel;

/**
 * Generalized  linear kernel on double[]. Provided a proper inner product matrix M, this kernel returns :
 * k(x, y) = x'*M*y
 * @author dpicard
 *
 */
public class GeneralizedDoubleLinear extends Kernel<double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6618610116348247480L;
	
	double[][] M;
	int size;
	
	public GeneralizedDoubleLinear(double[][] innerProduct)
	{
		if(innerProduct.length != innerProduct[0].length)
		{
			M = null;
			size = 0;
		}
		else
		{
			M = innerProduct;
			size = M.length;
		}
		
		
	}
	
	@Override
	public double valueOf(double[] t1, double[] t2) {
		
		if(t1.length != size && t2.length != size)
			return 0;
		
		double sum = 0;
		for(int i = 0 ; i < M.length; i++)
		{
			double xtM = 0;
			for(int j = 0 ; j < M[0].length; j++)
			{
				xtM += t1[j]*M[j][i];
			}
			sum += xtM*t2[i];
		}
		
		return sum;
	}

	@Override
	public double valueOf(double[] t1) {
		
		return valueOf(t1, t1);
	}

}
