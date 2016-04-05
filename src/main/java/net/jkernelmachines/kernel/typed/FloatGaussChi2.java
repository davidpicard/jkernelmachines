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
 * Gaussian Kernel on float[] that uses a Chi2 distance.
 * @author dpicard
 *
 */
public class FloatGaussChi2 extends Kernel<float[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1626829154456556731L;
	
	
	private double gamma = 0.1;
	private double eps = 10e-12;
	
	@Override
	public double valueOf(float[] t1, float[] t2) {
		double sum = 0.;
		double tmp = 0.;
		double min = 0.;
		for (int i = 0; i < Math.min(t1.length, t2.length); i++)
			//assume X and Y > 0
			if( (tmp = t1[i]+t2[i]) > eps)
			{
				min = t1[i]-t2[i];
				sum += (min*min) / tmp; //chi2
			}
		
		return Math.exp(-gamma * sum);
	}

	@Override
	public double valueOf(float[] t1) {
		return 1.0;
	}


	/**
	 * @return the sigma
	 */
	public double getGamma() {
		return gamma;
	}

	/**
	 * @param gamma inverse of std dev parameter
	 */
	public void setGamma(double gamma) {
		this.gamma = gamma;
	}

}
