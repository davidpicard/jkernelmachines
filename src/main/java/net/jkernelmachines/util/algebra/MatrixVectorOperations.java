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


/**
 * This class provides operations between matrices and vectors
 * @author picard
 *
 */
public class MatrixVectorOperations {

	/**
	 * Performs a matrix*vector multiplication
	 * @param A input matrix of size m*n
	 * @param x input vector of dimension n
	 * @return A*x of dimension n
	 */
	public static double[] rMul(final double[][] A, final double[] x) {
		int n = x.length;
		int m = A.length;
		if(A[0].length != n) {
			throw new ArithmeticException("Matrix Dimension must agree : "+A[0].length+", "+n);
		}
		double[] o = new double[m];
		
		return rMuli(o, A, x);
	}
	
	/**
	 * Performs a matrix*vector multiplication in place
	 * @param A input matrix of size m*n
	 * @param x input vector of dimension n
	 * @return A*x of dimension n
	 */
	public static double[] rMuli(double[] C, final double[][] A, final double[] x) {
		int n = x.length;
		int m = A.length;
		if(A[0].length != n) {
			throw new ArithmeticException("Matrix Dimension must agree : "+A[0].length+", "+n);
		}
		if(C.length != m) {
			throw new ArithmeticException("Matrix Dimension must agree : "+C.length+", "+m);
		}

		if(n > ThreadedMatrixVectorOperations.granularity) {
			return ThreadedMatrixVectorOperations.rMuli(C, A, x);
		}
		
		for(int i = 0 ; i < m ; i++) {
			C[i] = VectorOperations.dot(A[i], x);
		}
		
		return C;
	}
	
	/**
	 * Adds the tensor product of a vector to a matrix (usefull for covariance matrices)
	 * @param C output matrix, C = C +x*x'
	 * @param x input vector
	 * @return C
	 */
	public static double[][] addXXTrans(double[][] C, final double[] x) {
		int n = x.length;
		if(C.length != n || C[0].length!= n) {
			throw new ArithmeticException("Matrix dimensions must agree: "+C.length+", "+C[0].length+", "+n);
		}
		for(int i = 0 ; i < n ; i++) {
			for(int j = i ; j < n ; j++) {
				C[i][j] += x[i]*x[j];
				C[j][i] += x[i]*x[j];
			}
		}
		
		return C;
	}

	/**
	 * Computes the tensor (outer) product of two vectors
	 * @param x first vector
	 * @param y second vector
	 * @return x*y^T
	 */
	public static double[][] outer(final double[] x, final double[] y) {
		double[][] m = new double[x.length][y.length];
		
		for(int i = 0 ; i < m.length ; i++) {
			for(int j = 0 ; j < m[0].length ; j++) {
				m[i][j] = x[i]*y[j];
			}
		}
		
		return m;
	}
}
