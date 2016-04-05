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
package net.jkernelmachines.kernel.extra;

import net.jkernelmachines.kernel.Kernel;

/**
 * <p>
 * Kernel with a provided custom matrix.
 * </p>
 * <p>
 * The datatype of input space is integer relative to row/column indices. Therefore, the similarity
 * between elements i and j is matrix[i][j].
 * If i or j is not in the range of the matrix, 0 is returned.
 * </p>
 * @author dpicard
 *
 */
public class CustomMatrixKernel extends Kernel<Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5379932592270965091L;
	private double matrix[][];
	
	/**
	 * Constructor using the supplied Gram matrix.
	 * @param matrix the Gram matrix of underlying kernel function.
	 */
	public CustomMatrixKernel(double matrix[][])
	{
		this.matrix = matrix;
	}
	
	@Override
	public double valueOf(Integer t1, Integer t2) {
		if(t1 > matrix.length || t2 > matrix.length)
			return 0;
		return matrix[t1][t2];
	}

	@Override
	public double valueOf(Integer t1) {
		if(t1 > matrix.length)
			return 0.;
		return matrix[t1][t1];
	}

}
