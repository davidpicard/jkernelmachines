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

import static java.lang.Math.*;



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
		
		if(A.length < 64)
			qri_gramschmidt(Q, R, A);
		else
			qri_givens(Q, R, A);
	}
	

	/**
	 * Performs the in place QR decomposition of a symmetric matrix using Givens rotation matrices,
	 * such that Q is orthonormal and R is an upper triangular matrix:
	 * 
	 * A = Q*R
	 * 
	 * @param Q output matrix Q
	 * @param R output matrix R
	 * @param A input matrix A
	 */
	private static void qri_givens(double[][]Q, double[][] R, final double[][] m) {
		int n = m.length;

		// copy matrix
		for(int i = 0 ; i < n ; i++)
			for(int j = 0 ; j < n ; j++) {
				R[i][j] = m[i][j];
			}
		// init Q
		for(int i = 0 ; i < n ; i++)
			Q[i][i] = 1.;
		
		for(int i = 0 ; i < n ; i++) {
			for(int j = i+1 ; j < n ; j++) {
				double Rii = R[i][i];
				double Rji = R[j][i];
				double Rij = R[i][j];
				double Rjj = R[j][j];
				double Qii = Q[i][i];
				double Qij = Q[i][j];
				double Qji = Q[j][i];
				double Qjj = Q[j][j];
				
				double r = sqrt(Rii*Rii+Rji*Rji);
				double c = Rii/r;
				double s = -Rji/r;
				
				// apply rotation
				R[i][i] = r;
				R[j][i] = 0;
				R[j][j] = s * Rij + c * Rjj;
				R[i][j] = c*Rij - s*Rjj;
				Q[i][i] = c*Qii - s*Qji;
				Q[j][j] = s*Qij + c*Qjj;
				Q[i][j] = c*Qij - s*Qjj;
				Q[j][i] = s*Qii + c*Qji;
				
				
				for(int k = 0 ; k < n ; k++) {
					if(k == i || k == j)
						continue;
					double Rik = R[i][k];
					double Rjk = R[j][k];
					double Qik = Q[i][k];
					double Qjk = Q[j][k];
					
					R[i][k] = c*Rik - s*Rjk;
					R[j][k] = s*Rik + c*Rjk;
					Q[i][k] = c*Qik - s*Qjk;
					Q[j][k] = s*Qik + c*Qjk;
				}
			}
		}
		// transpose Q to get basis
		MatrixOperations.transi(Q);
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
	private static void qri_gramschmidt(double[][] Q, double[][] R, final double[][] A) {
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
		if(A.length < 64)
			return eig_qr(A);
		else
			return eig_givens(A, true);
	}

	/**
	 * Performs the eigen decomposition of a symmetric matrix:
	 * A = Q * L * Q'
	 * with Q orthonormal and L diagonal
	 * @param A input matrix
	 * @return an array of two matrices containing {Q, L}
	 */
	private static final double[][][] eig_givens(double[][] m, boolean prec) {
		int n = m.length;
		double[][] matrix;
		
		// init eigenvectors
		double[][] G;
		if(!prec)  {
			matrix = new double[n][n];
			for(int i = 0 ; i < n ; i++)
				for(int j = 0 ; j < n ; j++)
					matrix[i][j] = m[i][j];
			G = new double[n][n];
			for(int i = 0 ; i < n ; i++)
				G[i][i] = 1;
		}
		else
		{
			// QR preconditioning
			double[][][] QR = new double[2][n][n];
			qri_givens(QR[0], QR[1], m);
			matrix = MatrixOperations.mul(QR[1], QR[0]);
			G = MatrixOperations.transi(QR[0]);
		}
		
		while(true) {
			for(int i = 0 ; i < n-1 ; i++) {
				// search largest of diag for i
				int j = i+1;
				for(int id = i+1 ; id < n ; id++) {
					if(abs(matrix[i][id]) > abs(matrix[i][j]))
						j = id;
				}
				double Sii = matrix[i][i];
				double Sjj = matrix[j][j];
				double Sij = matrix[i][j];
				double Gii = G[i][i];
				double Gjj = G[j][j];
				double Gij = G[i][j];
				double Gji = G[j][i];
				//compute theta
				double theta = 0;
				if(Sii == Sjj)
					theta = PI / 4;
				else
					theta = atan2(2*Sij, (Sjj - Sii))/2;
				// compute givens rotation
				double c = cos(theta);
				double s = sin(theta);
				
				// apply givens rotation
				matrix[i][i] = c*c*Sii + s*s*Sjj - 2*c*s*Sij;
				matrix[j][j] = s*s*Sii + c*c*Sjj + 2*c*s*Sij;
				matrix[i][j] = (c*c-s*s)*Sij + s*c*(Sii - Sjj);
				matrix[j][i] = matrix[i][j];
				G[i][i] = c*Gii - s*Gji;
				G[i][j] = c*Gij - s*Gjj;
				G[j][i] = s*Gii + c*Gji;
				G[j][j] = s*Gij + c*Gjj;
				
				for(int k = 0 ; k < n ; k++) {
					double Sik = matrix[i][k];
					double Sjk = matrix[j][k];
					double Gik = G[i][k];
					double Gjk = G[j][k];
					
					if(k != i && k != j) {
						matrix[i][k] = c*Sik - s*Sjk;
						matrix[k][i] = matrix[i][k];
						matrix[j][k] = s*Sik + c*Sjk;
						matrix[k][j] = matrix[j][k];
						
						G[i][k] = c*Gik - s*Gjk;
						G[j][k] = s*Gik + c*Gjk;
					}
				}			
			}
			
			// check if done
			double max = 0;
			for(int i = 0 ; i < n-1 ; i++)
				for(int j = i+1 ; j < n ; j++)
					if(abs(matrix[i][j]) > max)
						max = abs(matrix[i][j]);
			if(max < 1e-15)
				break;
		}
		for(int i = 0 ; i < n ; i++)
			for(int j = i+1 ; j< n ; j++) {
				matrix[i][j] = 0;
				matrix[j][i] = 0;
			}
		return new double[][][] {G, matrix};
	}

	
	/**
	 * Performs the eigen decomposition of a symmetric matrix:
	 * A = Q * L * Q'
	 * with Q orthonormal and L diagonal
	 * @param A input matrix
	 * @return an array of two matrices containing {Q, L}
	 */
	private static double[][][] eig_qr(double[][] A) {
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
			qri_gramschmidt(QR[0], QR[1], Ak);
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