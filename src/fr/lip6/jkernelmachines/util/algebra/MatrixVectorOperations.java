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
		double[] o = new double[n];
		
		for(int i = 0 ; i < m ; i++) {
			o[i] = VectorOperations.dot(A[i], x);
		}
		
		return o;
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
}
