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
package net.jkernelmachines.util.algebra;

import net.jkernelmachines.threading.ThreadedMatrixOperator;

/**
 * This class provides multithreaded operations between matrices and vectors
 * 
 * @author picard
 * 
 */
public class ThreadedMatrixVectorOperations {

	public static int granularity = 1024;

	/**
	 * Performs a matrix*vector multiplication
	 * 
	 * @param A
	 *            input matrix of size m*n
	 * @param x
	 *            input vector of dimension n
	 * @return A*x of dimension n
	 */
	public static double[] rMul(final double[][] A, final double[] x) {
		int n = x.length;
		if (A[0].length != n) {
			throw new ArithmeticException("Matrix Dimension must agree : "
					+ A[0].length + ", " + n);
		}

		double[] o = new double[A.length];

		return rMuli(o, A, x);
	}

	/**
	 * Performs a matrix*vector multiplication in place
	 * 
	 * @param A
	 *            input matrix of size m*n
	 * @param x
	 *            input vector of dimension n
	 * @param C
	 *            output vector
	 * @return A*x of dimension n
	 */
	public static double[] rMuli(double[] C, final double[][] A,
			final double[] x) {
		int n = x.length;
		int m = A.length;
		if (A[0].length != n) {
			throw new ArithmeticException("Matrix Dimension must agree : "
					+ A[0].length + ", " + n);
		}
		if (C.length != m) {
			throw new ArithmeticException("Matrix Dimension must agree : "
					+ C.length + ", " + m);
		}

		final double[] o = C;
		ThreadedMatrixOperator tmo = new ThreadedMatrixOperator() {

			@Override
			public void doLines(double[][] matrix, int from, int to) {
				for (int i = from; i < to; i++) {
					o[i] = VectorOperations.dot(matrix[i], x);
				}
			}
		};

		tmo.getMatrix(A);

		return o;
	}

}
