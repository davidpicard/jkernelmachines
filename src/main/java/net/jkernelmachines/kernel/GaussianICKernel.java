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

import java.util.Map;

/**
 * Not so useful &lt; key,value &gt; based caching method for Gaussian kernels
 * @author picard
 *
 * @param <S>
 * @param <T>
 */
public class GaussianICKernel<S, T> extends GaussianKernel<S> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -8504726802546629864L;


	private double[][] matrix;
	private Map<S, Integer> map;
	private double gamma = 1.0;
	
	/**
	 * Constructor using underlying indexed kernel k, supposed to be Gaussian.
	 * @param k the underlying kernel
	 */
	public GaussianICKernel(IndexedCacheKernel<S, T> k) {
		
		this.map = k.getMap();
		double[][] m = k.getCacheMatrix();
		
		matrix = new double[m.length][m.length];
		for(int x = 0 ; x < m.length ; x++)
		for(int y = 0 ; y < m.length ; y++) {
			double tmp = - Math.log(m[x][y]);
			if(tmp < 1e-4)
				matrix[x][y] = 0;
			else
				matrix[x][y] = tmp;
		}
		
	}
	
	@Override
	public double distanceValueOf(S t1, S t2) {
		assert(map.containsKey(t1) && map.containsKey(t2));
		
//		{
//			System.err.println("<"+t1+","+t2+"> not in matrix !!!");
//			return 0;
//		}
		int id1 = map.get(t1);
		int id2 = map.get(t2);
		
		return ((double)(float)matrix[id1][id2]);
	}

	@Override
	public double getGamma() {
		return gamma;
	}

	@Override
	public void setGamma(double gamma) {
		this.gamma = gamma;
	}

	@Override
	public double valueOf(S t1, S t2) {
		double tmp = -gamma*distanceValueOf(t1, t2);
		if(tmp >= 10) //num cleaning
			return 0;
		return Math.exp(-gamma*distanceValueOf(t1, t2));
	}

	@Override
	public double valueOf(S t1) {
		return 1.0;
	}


}
