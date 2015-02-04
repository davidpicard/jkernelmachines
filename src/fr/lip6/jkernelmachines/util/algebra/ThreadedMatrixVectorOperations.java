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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

import fr.lip6.jkernelmachines.threading.ThreadPoolServer;

/**
 * This class provides multithreaded operations between matrices and vectors
 * 
 * @author picard
 * 
 */
public class ThreadedMatrixVectorOperations {

	public static int granularity = 8;
	
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
		int m = A.length;
		if (A[0].length != n) {
			throw new ArithmeticException("Matrix Dimension must agree : "
					+ A[0].length + ", " + n);
		}

		final double[] o = new double[n];

		ThreadPoolExecutor exec = ThreadPoolServer.getThreadPoolExecutor();
		List<Future<Object>> futures = new ArrayList<Future<Object>>();

		for (int ii = 0; ii < m; ii++) {
			final int i = ii;
			futures.add(exec.submit(new Callable<Object>() {

				@Override
				public Object call() throws Exception {
					o[i] = VectorOperations.dot(A[i], x);
					return null;
				}
			}));

		}

		for (Future<Object> f : futures) {
			try {
				f.get();
			} catch (Exception e) {
				throw new ArithmeticException(
						"Threaded algebraic operations unavailable");
			}
		}

		return o;
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
		
		ThreadPoolExecutor exec = ThreadPoolServer.getThreadPoolExecutor();
		List<Future<Object>> futures = new ArrayList<Future<Object>>();
		final double[] o = C;
		
		for (int ii = 0; ii < m; ii++) {
			final int i = ii;
			futures.add(exec.submit(new Callable<Object>() {

				@Override
				public Object call() throws Exception {
					o[i] = VectorOperations.dot(A[i], x);
					return null;
				}
			}));

		}

		for (Future<Object> f : futures) {
			try {
				f.get();
			} catch (Exception e) {
				throw new ArithmeticException(
						"Threaded algebraic operations unavailable");
			}
		}

		return C;
	}

}
