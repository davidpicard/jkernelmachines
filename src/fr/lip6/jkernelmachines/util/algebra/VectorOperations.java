/**
    This file is part of JkernelMachines.

    JkernelMachines is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    JkernelMachines is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with JkernelMachines.  If not, see <http://www.gnu.org/licenses/>.

    Copyright David Picard - 2013

*/
package fr.lip6.jkernelmachines.util.algebra;

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
	 * @return C
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
	
	public static double[] prod(final double[] A, final double[] B) {
		double[] out = new double[A.length];
		prodi(out, A, B);
		return out;
	}
	
	public static double[] prodi(double[] C, final double[] A, final double[] B) {
		int packed = 2 * (A.length / 2);
		int l = 0;
		// packed operations
		for(l = 0 ; l < packed ; l += 2) {
			C[l] = A[l]*B[l];
			C[l+1] = A[l+1]*B[l];
		}
		// remaining operations
		for(; l < A.length ; l++) {
			C[l] = A[l]*B[l];
		}
		
		return C;
	}

	/**
	 * Computes the dot product between to double arrays
	 * @param a first array
	 * @param b second array
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
	 * @return
	 */
	public static double d2p2(final double[] A, final double[] B) {
		return n2p2(A) + n2p2(B) - 2*dot(A, B);
	}
}