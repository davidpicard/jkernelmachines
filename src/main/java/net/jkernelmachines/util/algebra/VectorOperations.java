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
 * This class provides level 1 (vector) basic linear algebra operations.
 * @author picard
 *
 */
public class VectorOperations {
	
	/**
	 * Performs a linear combination of 2 vectors and store the result in a newly allocated array C:
	 * C = A + lambda * B
	 * @param A first vector
	 * @param lambda weight of the second vector
	 * @param B second vector
	 * @return A + lambda * B
	 */
	public static double[] add(final double[] A, final double lambda, final double[] B) {
		double[] out = new double[A.length];
		
		addi(out, A, lambda, B);
		
		return out;
	}
	
	/**
	 * Performs a linear combination of 2 vectors and store the result in an already allocated array C:
	 * C = A + lambda * B
	 * @param C output vector
	 * @param A first vector
	 * @param lambda weight of the second vector
	 * @param B second vector
	 * @return C vector
	 */
	public static double[] addi(double[] C, final double[] A, final double lambda, final double[] B) {	
		int packed = 2 * (A.length / 2);
		int l = 0;
		// packed operations
		for(l = 0 ; l < packed ; l += 2) {
			C[l] = A[l] + lambda*B[l];
			C[l+1] = A[l+1] + lambda*B[l+1];
		}
		// remaining operations
		for(; l < A.length ; l++) {
			C[l] = A[l] + lambda*B[l];
		}
		return C;
	}
	
	/**
	 * Multiply a given double array by a constant double:
	 * C = lambda * A
	 * @param A the input array
	 * @param lambda the constant
	 * @return a new array containing the result
	 */
	public static double[] mul(final double[] A, final double lambda) {
		double[] out = new double[A.length];
		
		muli(out, A, lambda);
		
		return out;
	}
	
	/**
	 * Multiply a given double array by a constant double:
	 * C = lambda * A
	 * @param A the input array
	 * @param lambda the constant
	 * @param C output
	 * @return a new array containing the result
	 */
	public static double[] muli(double[] C, final double[] A, final double lambda) {
		int packed = 2 * (A.length / 2);
		int l = 0;
		// packed operations
		for(l = 0 ; l < packed ; l += 2) {
			C[l] = A[l]*lambda;
			C[l+1] = A[l+1]*lambda;
		}
		// remaining operations
		for(; l < A.length ; l++) {
			C[l] = A[l]*lambda;
		}
		
		return C;
	}
	
	/**
	 * Performs the component wise product between 2 vectors
	 * @param A first array
	 * @param B second array
	 * @return an array containing the compoentn wise product
	 */
	public static double[] prod(final double[] A, final double[] B) {
		double[] out = new double[A.length];
		prodi(out, A, B);
		return out;
	}
	
	/**
	 * Performs the in place component wise product between 2 vectors
	 * @param C the output array
	 * @param A first array
	 * @param B second array
	 * @return C
	 */
	public static double[] prodi(double[] C, final double[] A, final double[] B) {
		int packed = 2 * (A.length / 2);
		int l = 0;
		// packed operations
		for(l = 0 ; l < packed ; l += 2) {
			C[l] = A[l]*B[l];
			C[l+1] = A[l+1]*B[l+1];
		}
		// remaining operations
		for(; l < A.length ; l++) {
			C[l] = A[l]*B[l];
		}
		
		return C;
	}

	/**
	 * Computes the dot product between to double arrays
	 * @param A first array
	 * @param B second array
	 * @return the dot product between A and B
	 */
	public static double dot(final double[] A, final double[] B) {
		
		double sum = 0;
		int packed = 2 * (A.length / 2);
		int k = 0;
		//packed operations
		for(k = 0 ; k < packed ; k+=2) {
			sum += A[k]*B[k] + A[k+1]*B[k+1];
		}
		// remaining operations
		for(; k < A.length; k++) {
			sum += A[k]*B[k];
		}
		
		return sum;
	}
	
	/**
	 * Computes the l2 norm of a double array
	 * @param A the array
	 * @return the l2 norm of A
	 */
	public static double n2(final double[] A) {
		return Math.sqrt(dot(A, A));
	}
	
	/**
	 * Computes the squared l2 norm of a double array
	 * @param A the array
	 * @return the squared l2 norm of A
	 */
	public static double n2p2(final double[] A) {
		return dot(A, A);
	}
	
	/**
	 * Computes the square euclidean distance between 2 double arrays
	 * @param A first array
	 * @param B second array
	 * @return squared euclidean distance
	 */
	public static double d2p2(final double[] A, final double[] B) {
		return n2p2(A) + n2p2(B) - 2*dot(A, B);
	}
}