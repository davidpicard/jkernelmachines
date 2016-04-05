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
				for(int i = from ; i < to ; i++) {
					o[i] = VectorOperations.dot(matrix[i], x);
				}
			}
		};
		
		tmo.getMatrix(A);


		return o;
	}

}
