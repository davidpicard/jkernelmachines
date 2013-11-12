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
package fr.lip6.jkernelmachines.util.algebra.ejml;

import java.util.Arrays;

import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.factory.EigenDecomposition;

import fr.lip6.jkernelmachines.util.algebra.ThreadedMatrixOperations;

/**
 * Wrapper class that uses EJML for Matrix ops, with the API of jkms
 * @author picard
 *
 */
public class EJMLMatrixOperations {

	/**
	 * Performs the eigen decomposition of a symmetric matrix:
	 * A = Q * L * Q'
	 * with Q orthonormal and L diagonal
	 * @param A input matrix
	 * @return an array of two matrices containing {Q, L}
	 */
	public static double[][][] eig(double[][] A) {
		int n = A.length;
		EigenDecomposition<DenseMatrix64F> dec = DecompositionFactory.eig(n, true, true);
		DenseMatrix64F G_ejml = new DenseMatrix64F(A);
		dec.decompose(G_ejml);
		double[][] lambda = new double[n][n];
		double[][] U = new double[n][];
		for(int i = 0 ; i < n ; i++) {
			lambda[i][i] = dec.getEigenvalue(i).getMagnitude();
			DenseMatrix64F u = dec.getEigenVector(i);
			U[i] = Arrays.copyOf(u.data, n);
			
		}
		return new double[][][]{ThreadedMatrixOperations.transi(U), lambda};
	}
}
