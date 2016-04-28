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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

import net.jkernelmachines.threading.ThreadPoolServer;

/**
 * This class provides multithreaded basic linear algebra operations on matrices.
 * @author picard
 *
 */
public class ThreadedMatrixOperations {
	/**
	 * threshold under which single-threaded ops are used
	 */
	public static int granularity = 65;
	
	/**
	 * Computes the transposed matrix of a matrix
	 * @param A the input matrix
	 * @return a newly allocated matrix containing the transpose of A
	 */
	public static double[][] trans(final double[][] A) {

		if(A.length < granularity)
			return MatrixOperations.trans(A);
		
		final double[][] out = new double[A[0].length][A.length];
		
		ThreadPoolExecutor exec = ThreadPoolServer.getThreadPoolExecutor();
		List<Future<Object>> futures = new ArrayList<Future<Object>>();
		
		for(int jj = 0 ; jj < A.length ; jj++) {
			final int j = jj;
			futures.add(exec.submit(new Callable<Object>() {

				@Override
				public Object call() {
					for(int i = 0 ; i < A[0].length ; i++) {
							out[i][j] = A[j][i];
					}
					return null;
				}
			}));
		}

		for(Future<Object> f : futures)
			try {
				f.get();
			} catch (InterruptedException e) {
				e.printStackTrace();
				return MatrixOperations.trans(A);
			} catch (ExecutionException e) {
				e.printStackTrace();
				return MatrixOperations.trans(A);
			}
		
		return out;
	}
	
	/**
	 * Computes the transposed matrix of a symmetric matrix in place
	 * @param A the input matrix to transpose
	 * @return A transpose
	 */
	public static double[][] transi(double[][] A) {
		
		if(A.length < granularity)
			return MatrixOperations.transi(A);
		
		if(!MatrixOperations.isSquare(A))
			throw new ArithmeticException("Matrix must be square.");

		ThreadPoolExecutor exec = ThreadPoolServer.getThreadPoolExecutor();
		List<Future<Object>> futures = new ArrayList<Future<Object>>();
		final double[][] m = A;
		
		for(int ii = 0 ; ii < A.length ; ii++) {
			final int i = ii;
			futures.add(exec.submit(new Callable<Object>() {
				double tmp;
				@Override
				public Object call() {
					for(int j = i+1 ; j < m[0].length; j++) {
						tmp = m[i][j];
						m[i][j] = m[j][i];
						m[j][i] = tmp;
					}
					return null;
				}
			}));
		}

		for(Future<Object> f : futures)
			try {
				f.get();
			} catch (InterruptedException e) {
				e.printStackTrace();
				return MatrixOperations.transi(A);
			} catch (ExecutionException e) {
				e.printStackTrace();
				return MatrixOperations.transi(A);
			}
		return m;
	}
	
	/**
	 * Performs the matrix multiplication between two double matrices
	 * C = A * B
	 * @param A first matrix
	 * @param B second matrix
	 * @return a newly allocated matrix C
	 * @throws ArithmeticException if dimensions not compatible
	 */
	public static double[][] mul(final double[][] A, final double[][] B) throws ArithmeticException {
		
		final int n, m, p;
		m = A.length;
		n = B[0].length;
		p = A[0].length;
		
		if(n < granularity || m < granularity || p < granularity)
			return MatrixOperations.mul(A, B);
		
		if(p != B.length) {
			throw new ArithmeticException("Matrix dimensions must agree.");
		}
		
		final double[][] out = new double[m][n];
		
		ThreadPoolExecutor exec = ThreadPoolServer.getThreadPoolExecutor();
		List<Future<Object>> futures = new ArrayList<Future<Object>>();
		
		for(int ii = 0 ; ii < m ; ii++) {
			final int i = ii;
			futures.add(exec.submit(new Callable<Object>() {

				@Override
				public Object call() throws Exception {
					for(int j = 0 ; j < n ; j++) {
						double sum = 0;
						for(int k = 0 ; k < p ; k++) {
							sum += A[i][k]*B[k][j];
						}
						out[i][j] = sum;
					}
					return null;
				}}));
			
		}
		
		for(Future<Object> f : futures)
			try {
				f.get();
			} catch (InterruptedException e) {
				e.printStackTrace();
				return MatrixOperations.mul(A, B);
			} catch (ExecutionException e) {
				e.printStackTrace();
				return MatrixOperations.mul(A, B);
			}
		
		return out;
	}
	
	
	/**
	 * Performs the matrix multiplication between two double matrices
	 * C = A * B
	 * @param A first matrix
	 * @param B second matrix
	 * @param C the result matrix
	 * @return matrix C
	 * @throws ArithmeticException if dimensions not compatible
	 */
	public static double[][] muli(double[][]C, final double[][] A, final double[][] B) throws ArithmeticException {
		
		final int n, m, p;
		m = A.length;
		n = B[0].length;
		p = A[0].length;
		final double[][] out = C;
		
		if(n < granularity || m < granularity || p < granularity)
			return MatrixOperations.muli(C, A, B);
		
		if(p != B.length) {
			throw new ArithmeticException("Matrix dimensions must agree.");
		}
		
		ThreadPoolExecutor exec = ThreadPoolServer.getThreadPoolExecutor();
		List<Future<Object>> futures = new ArrayList<Future<Object>>();
		
		for(int ii = 0 ; ii < m ; ii++) {
			final int i = ii;
			futures.add(exec.submit(new Callable<Object>() {

				@Override
				public Object call() throws Exception {
					for(int j = 0 ; j < n ; j++) {
						double sum = 0;
						for(int k = 0 ; k < p ; k++) {
							sum += A[i][k]*B[k][j];
						}
						out[i][j] = sum;
					}
					return null;
				}}));
			
		}
		
		for(Future<Object> f : futures)
			try {
				f.get();
			} catch (InterruptedException e) {
				e.printStackTrace();
				return MatrixOperations.muli(C, A, B);
			} catch (ExecutionException e) {
				e.printStackTrace();
				return MatrixOperations.muli(C, A, B);
			}
		
		return out;
	}
	

	/**
	 * Computes the transpose multiplication between two matrices : 
	 * C = A' * B
	 * @param A first matrix
	 * @param B second matrix
	 * @return C = A' * B
	 */
	public static double[][] transMul(final double[][] A, final double[][] B) {
		
		final int n, m, p;
		m = A.length;
		n = B[0].length;
		p = A[0].length;

		if(n < granularity || m < granularity || p < granularity)
			return MatrixOperations.transMul(A, B);
		
		if(p != B.length) {
			throw new ArithmeticException("Matrix dimensions must agree.");
		}
		
		final double[][] out = new double[m][n];
		
		if(p != B.length) {
			throw new ArithmeticException("Matrix dimensions must agree.");
		}
		
		ThreadPoolExecutor exec = ThreadPoolServer.getThreadPoolExecutor();
		List<Future<Object>> futures = new ArrayList<Future<Object>>();
		
		for(int ii = 0 ; ii < m ; ii++) {
			final int i = ii;
			futures.add(exec.submit(new Callable<Object>() {

				@Override
				public Object call() throws Exception {
					for(int j = 0 ; j < n ; j++) {
						double sum = 0;
						for(int k = 0 ; k < p ; k++) {
							sum += A[k][i]*B[k][j];
						}
						out[i][j] = sum;
					}
					return null;
				}}));
			
		}
		
		for(Future<Object> f : futures)
			try {
				f.get();
			} catch (InterruptedException e) {
				e.printStackTrace();
				return MatrixOperations.transMul(A, B);
			} catch (ExecutionException e) {
				e.printStackTrace();
				return MatrixOperations.transMul(A, B);
			}
		
		return out;		
	}
	
	/**
	 * Computes the transpose multiplication between two matrices : 
	 * C = A' * B
	 * @param A first matrix
	 * @param B second matrix
	 * @param C output matrix
	 * @return C = A' * B
	 */
	public static double[][] transMuli(double[][] C, final double[][] A, final double[][] B) {

		final int n, m, p;
		m = A.length;
		n = B[0].length;
		p = A[0].length;
		final double[][] out = C;

		if(n < granularity || m < granularity || p < granularity)
			return MatrixOperations.transMuli(C, A, B);
		
		if(p != B.length) {
			throw new ArithmeticException("Matrix dimensions must agree.");
		}
		
		ThreadPoolExecutor exec = ThreadPoolServer.getThreadPoolExecutor();
		List<Future<Object>> futures = new ArrayList<Future<Object>>();
		
		for(int ii = 0 ; ii < m ; ii++) {
			final int i = ii;
			futures.add(exec.submit(new Callable<Object>() {

				@Override
				public Object call() throws Exception {
					for(int j = 0 ; j < n ; j++) {
						double sum = 0;
						for(int k = 0 ; k < p ; k++) {
							sum += A[k][i]*B[k][j];
						}
						out[i][j] = sum;
					}
					return null;
				}}));
			
		}
		
		for(Future<Object> f : futures)
			try {
				f.get();
			} catch (InterruptedException e) {
				e.printStackTrace();
				return MatrixOperations.transMuli(C, A, B);
			} catch (ExecutionException e) {
				e.printStackTrace();
				return MatrixOperations.transMuli(C, A, B);
			}
		
		return out;		
	}
}

