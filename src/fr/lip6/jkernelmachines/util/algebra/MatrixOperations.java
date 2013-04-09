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
 * This class provides basic linear algebra operations on matrices.
 * @author picard
 *
 */
public class MatrixOperations {
	
	
	/**
	 * Tests if a matrix is square
	 * @param A the input matrix
	 * @return true if A is n*n, false else
	 */
	public static boolean isSquare(final double[][] A) {
		if(A.length != A[0].length) {
			return false;
		}
		return true;
	}
	
	/**
	 * Checks if a matrix is symmetric
	 * @param A the matrix
	 * @return true if A is symmetric, false else
	 */
	public static boolean isSymmetric(final double[][] A) {
		if(!isSquare(A))
			throw new ArithmeticException("Matrix must be square");
		
		for(int i = 0 ; i < A.length; i++) {
			for(int j = i+1 ; j < A[0].length ; j++) {
				if(A[i][j] != A[j][i])
					return false;
			}
		}
		return true;
	}
	
	/**
	 * Computes the transposed matrix of a matrix
	 * @param A the input matrix
	 * @return a newly allocated matrix containing the transpose of A
	 */
	public static double[][] trans(final double[][] A) {
		double[][] out = new double[A[0].length][A.length];
		
		for(int j = 0 ; j < A.length; j++) {
			for(int i = 0 ; i < A[0].length ; i++) {
				out[i][j] = A[j][i];
			}
		}
		
		return out;
	}
	
	/**
	 * Computes the transposed matrix of a symmetric matrix in place
	 * @param A the input matrix to transpose
	 * @return A transpose
	 */
	public static double[][] transi(double[][] A) {
		double tmp;
		
		if(!isSquare(A))
			throw new ArithmeticException("Matrix must be square.");

		for(int i = 0 ; i < A.length ; i++) { 
			for(int j = i+1 ; j < A[0].length; j++) {
				tmp = A[i][j];
				A[i][j] = A[j][i];
				A[j][i] = tmp;
			}
		}
		
		return A;
	}
	
	/**
	 * Performs the matrix multiplication between two double matrices
	 * C = A * B
	 * @param A first matrix
	 * @param B second matrix
	 * @return a newly allocated matrix C
	 * @throws ArithmeticException
	 */
	public static double[][] mul(final double[][] A, final double[][] B) throws ArithmeticException {
		
		int n, m, p;
		m = A.length;
		n = B[0].length;
		p = A[0].length;
		
		if(p != B.length) {
			throw new ArithmeticException("Matrix dimensions must agree.");
		}
		
		double[][] out = new double[m][n];
		
		for(int i = 0 ; i < m ; i++) {
			for(int j = 0 ; j < n ; j++) {
				double sum = 0;
				for(int k = 0 ; k < p ; k++) {
					sum += A[i][k]*B[k][j];
				}
				out[i][j] = sum;
			}
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
	 * @throws ArithmeticException
	 */
	public static double[][] muli(double[][]C, final double[][] A, final double[][] B) throws ArithmeticException {
		
		int n, m, p;
		m = A.length;
		n = B[0].length;
		p = A[0].length;
		
		if(p != B.length) {
			throw new ArithmeticException("Matrix dimensions must agree.");
		}
		for(int i = 0 ; i < m ; i++) {
			for(int j = 0 ; j < n ; j++) {
				double sum = 0;
				for(int k = 0 ; k < p ; k++) {
					sum += A[i][k]*B[k][j];
				}
				C[i][j] = sum;
			}
		}
		
		return C;
	}
	
	/**
	 * Computes the transpose multiplication between two matrices : 
	 * C = A' * B
	 * @param A first matrix
	 * @param B second matrix
	 * @return C = A' * B
	 */
	public static double[][] transMul(double[][] A, double[][] B) {
		
		int n, m, p;
		m = A[0].length;
		n = B[0].length;
		p = A.length;
		
		if(p != B.length) {
			throw new ArithmeticException("Matrix dimensions must agree.");
		}
		
		double[][] out = new double[m][n];
		
		for(int i = 0 ; i < m ; i++) {
			for(int j = 0 ; j < n ; j++) {
				double sum = 0;
				for(int k = 0 ; k < p ; k++) {
					sum += A[k][i]*B[k][j];
				}
				out[i][j] = sum;
			}
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
	public static double[][] transMuli(double[][] C, final double[][] A, final double[][] B) {
		
		int n, m, p;
		m = A[0].length;
		n = B[0].length;
		p = A.length;
		
		if(p != B.length) {
			throw new ArithmeticException("Matrix dimensions must agree.");
		}
		
		for(int i = 0 ; i < m ; i++) {
			for(int j = 0 ; j < n ; j++) {
				double sum = 0;
				for(int k = 0 ; k < p ; k++) {
					sum += A[k][i]*B[k][j];
				}
				C[i][j] = sum;
			}
		}
		
		return C;
		
	}
	
	/**
	 * Performs the QR decomposition of a symmetric matrix, such that Q is orthonormal 
	 * and R is an upper triangular matrix:
	 * 
	 * A = Q*R
	 *
	 * @param A input matrix
	 * @return an newly allocated array of two matrices containing {Q, R}
	 */
	public static double[][][] qr(double[][] A) {
		
		double[][] Q = new double[A.length][A[0].length];
		double[][] R = new double[A.length][A[0].length];
		
		qri(Q, R, A);
		
		double[][][] out = {Q, R};
		return out;
	}
	
	/**
	 * Performs the in place QR decomposition of a symmetric matrix, such that Q is orthonormal 
	 * and R is an upper triangular matrix:
	 * 
	 * A = Q*R
	 * 
	 * @param Q output matrix Q
	 * @param R output matrix R
	 * @param A input matrix A
	 */
	public static void qri(double[][] Q, double[][] R, final double[][] A) {
		if(!isSquare(A))
			throw new ArithmeticException("Matrix must be square");

		//copy first 
		double[] e = new double[A.length];
		for(int l = 0 ; l < A.length ; l++)
			e[l] = A[l][0];
		double n = VectorOperations.n2(e);
		VectorOperations.muli(Q[0], e, 1./n);
		R[0][0] = n;
		
		//Gram-schmidt
		for(int k = 1 ; k < A[0].length ; k++) {
			//copy
			for(int l = 0 ; l < A.length ; l++){
				e[l] = A[l][k];
			}
			
			for(int l = 0 ; l < k ; l++) {
				double p = VectorOperations.dot(e, Q[l]);
				R[l][k] = p;
				VectorOperations.addi(e, e, -p, Q[l]);				
			}
			double p = VectorOperations.n2(e);
			R[k][k] = p;
			if(p != 0) {
				VectorOperations.muli(Q[k], e, 1./p);
			}
			else {
				VectorOperations.muli(Q[k], e, 0.);
			}
		}		
		
		// transpose to have column vectors in Q
		transi(Q);
	}
	
	/**
	 * Performs the eigen decomposition of a symmetric matrix:
	 * A = Q * L * Q'
	 * with Q orthonormal and L diagonal
	 * @param A input matrix
	 * @return an array of two matrices containing {Q, L}
	 */
	public static double[][][] eig(double[][] A) {
		double[][] Q = null;
		
		if(!isSymmetric(A))
			throw new ArithmeticException("Matrix must be symmetric");
		
		double[][][] QR = qr(A);
		double[][] Ak = new double[A.length][A[0].length];
		double err = 0;
		do{
			//store eigenvectors Qk'*Q
			if(Q == null){
				Q = trans(QR[0]);
			}
			else{
				Q = transMul(QR[0], Q);
			}
			
			//R*Q
			muli(Ak, QR[1], QR[0]);

			// absolute error thanks to Gerschgorin Kreis
			double errt = 0;
			for(int i = 0 ; i < Ak.length ; i++){
				double sum = 0;
				for(int j = 0 ; j < Ak[0].length ; j++){
					if(i != j)
						sum += Math.abs(Ak[i][j]);
				}
				if(sum > errt)
					errt = sum;
			}
			if(Math.abs(err - errt) < 1e-12)
				break;
			err = errt;
			// next iteration
			qri(QR[0], QR[1], Ak);
		}
		while(err > 1e-12);
		
		//store eigenvalues
		double[][] eig = new double[A.length][A.length];
		for(int i = 0 ; i < Ak.length ; i++)
			eig[i][i] = Ak[i][i];
		
		// eigen vectors are Q'
		double[][][] out = {transi(Q), eig};
		return out;
	}

}