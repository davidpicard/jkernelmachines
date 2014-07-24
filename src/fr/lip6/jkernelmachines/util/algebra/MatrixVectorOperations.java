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

    Copyright David Picard - 2014

*/
package fr.lip6.jkernelmachines.util.algebra;


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
		
		if(m > ThreadedMatrixVectorOperations.granularity && n > ThreadedMatrixVectorOperations.granularity) {
			return ThreadedMatrixVectorOperations.rMul(A, x);
		}
		
		double[] o = new double[m];
		
		for(int i = 0 ; i < m ; i++) {
			o[i] = VectorOperations.dot(A[i], x);
		}
		
		return o;
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

		if(m > ThreadedMatrixVectorOperations.granularity && n > ThreadedMatrixVectorOperations.granularity) {
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
