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

import static fr.lip6.jkernelmachines.util.algebra.VectorOperations.addi;
import static fr.lip6.jkernelmachines.util.algebra.VectorOperations.dot;
import static fr.lip6.jkernelmachines.util.algebra.VectorOperations.n2;
import static java.lang.Math.PI;
import static java.lang.Math.abs;
import static java.lang.Math.atan2;
import static java.lang.Math.cos;
import static java.lang.Math.random;
import static java.lang.Math.signum;
import static java.lang.Math.sin;
import static java.lang.Math.sqrt;

import java.lang.reflect.Method;
import java.util.Arrays;

import fr.lip6.jkernelmachines.util.DebugPrinter;



/**
 * This class provides basic linear algebra operations on matrices.
 * @author picard
 *
 */
public class MatrixOperations {
	
	public final static double num_prec = 1e-14;
	
	/* try to load ejml wrapper if present, to accelerate matrix eig */
	static Method ejml_eig = null;
	static Method ejml_inv = null;
	static {
		try {
			Class.forName("org.ejml.data.DenseMatrix64F"); // check if ejml is there
			Class<?> emjl_ops = Class.forName("fr.lip6.jkernelmachines.util.algebra.ejml.EJMLMatrixOperations");
			ejml_eig = emjl_ops.getDeclaredMethod("eig", double[][].class);
			ejml_inv = emjl_ops.getDeclaredMethod("inv", double[][].class);
		} catch (Exception e) {
			ejml_eig = null;
			ejml_inv = null;
			if(DebugPrinter.DEBUG_LEVEL > 1) {
				System.err.println("Warning ejml not present, some operations will be slow");
			}
			if(DebugPrinter.DEBUG_LEVEL > 3) {
				e.printStackTrace();
			}
		}
	}
	
	
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
	
	public static boolean isTriDiagonal(final double[][] A) {
		int n = A.length;
		if (!isSquare(A))
			return false;
		for(int i = 0 ; i < n ; i++) {
			for(int j = 0 ; j < n ; j++) {
				if((i != j-1 && i != j && i != j+1) && abs(A[i][j]) > 1e-15)
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
	 * Computes the transposed matrix of a matrix in place
	 * @param C the output matrix
	 * @param A the input matrix
	 * @return a newly allocated matrix containing the transpose of A
	 */
	public static double[][] transi(double[][] C, final double[][] A) {
		if(C.length != A[0].length || C[0].length != A.length) {
			throw new ArithmeticException("Matrix dimension must agree: "+A.length+"x"+A[0].length+" != "+C[0].length+"x"+C.length);
		}
		
		for(int j = 0 ; j < A.length; j++) {
			for(int i = 0 ; i < A[0].length ; i++) {
				C[i][j] = A[j][i];
			}
		}
		
		return C;
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
		
		if(m > ThreadedMatrixOperations.granularity && n > ThreadedMatrixOperations.granularity) {
			return ThreadedMatrixOperations.mul(A, B);
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

		if(m > ThreadedMatrixOperations.granularity && n > ThreadedMatrixOperations.granularity) {
			return ThreadedMatrixOperations.muli(C, A, B);
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
	 * Performs the Givens rotation that nullifies component (i,j) of a matrix, accumulated on previous Rotation matrices
	 * @param matrix the input/output matrix
	 * @param G the input/output rotation matrix
	 * @param i component one the lines
	 * @param j component on the columns
	 * @return matrix
	 */
	public static double[][] givensRoti(double[][] matrix, double[][] G, int i , int j) {
		int n = matrix.length;
		
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
			if(k != i && k != j) {
				double Sik = matrix[i][k];
				double Sjk = matrix[j][k];
				double Gik = G[i][k];
				double Gjk = G[j][k];
				matrix[i][k] = c*Sik - s*Sjk;
				matrix[k][i] = matrix[i][k];
				matrix[j][k] = s*Sik + c*Sjk;
				matrix[k][j] = matrix[j][k];
				
				G[i][k] = c*Gik - s*Gjk;
				G[j][k] = s*Gik + c*Gjk;
			}
		}		
		return matrix;
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
	 * @param m input matrix m
	 */
	public static void qri_givens(double[][]Q, double[][] R, final double[][] m) {
		int n = m.length;

		// copy matrix
		for(int i = 0 ; i < n ; i++)
			for(int j = 0 ; j < n ; j++) {
				R[i][j] = m[i][j];
			}
		// init Q
		for(int i = 0 ; i < n ; i++) {
			Arrays.fill(Q[i],  0);
			Q[i][i] = 1.;
		}
		
		for(int i = 0 ; i < n-1 ; i++) {
			for(int j = i+1 ; j < n ; j++) {
				if(abs(R[i][j]) < 1e-14) {
					R[i][j] = 0;
					continue;
				}
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
		
		// num cleaning
		for(int i = 0 ; i < n ; i++) {
			for(int j = 0 ; j < n ; j++) {
				if(abs(Q[i][j]) < 1e-15)
					Q[i][j] = 0;
				if(abs(R[i][j]) < 1e-15)
					R[i][j] = 0;
			}
		}
		
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
	public static void qri_gramschmidt(double[][] Q, double[][] R, final double[][] A) {
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
	 * Compute the inverse matrix
	 * @param A input matrix
	 * @return the inverse of A if possible
	 */
	public static double[][] inv(final double[][] A) {
		int n = A.length;
		if( n > 65 && ejml_inv != null) {
			try {
				return (double[][]) ejml_inv.invoke(null,  new Object[]{A});
			}
			catch (Exception e) {
				if(DebugPrinter.DEBUG_LEVEL > 3)
					e.printStackTrace();
			}
		}
		// fallback to our implementation
		double[][][] ei = eig(A);
		double[][] u = ei[0];
		double[][] l = ei[1];
		
		for(int i = 0 ; i < l.length ; i++) {
			if(l[i][i] <= -num_prec)
				return null;
			if(abs(l[i][i]) > num_prec) {
				l[i][i] = 1/l[i][i];
			}
			else {
				l[i][i] = 0;
			}
		}
		return mul(u, mul(l, trans(u)));
	}
	
	
	
	/**
	 * Performs the eigen decomposition of a symmetric matrix:
	 * A = Q * L * Q'
	 * with Q orthonormal and L diagonal
	 * @param A input matrix
	 * @return an array of two matrices containing {Q, L}
	 */
	public static double[][][] eig(double[][] A) {
		// try ejml first if present
		if(A.length > 65 && ejml_eig != null)
			try {
				return (double[][][]) ejml_eig.invoke(null, new Object[]{A});
			} catch (Exception e) {
				if(DebugPrinter.DEBUG_LEVEL > 3)
					e.printStackTrace();
			}
		// fallback to our implementation
		if(DebugPrinter.DEBUG_LEVEL > 3)
			System.err.println("fallback to eig_jacobi");
		return eig_jacobi(A, true);
	}
	
	/**
	 * Performs the eigen decomposition of a symmetric matrix:
	 * A = Q * L * Q'
	 * with Q orthonormal and L diagonal
	 * @param m input matrix
	 * @return an array of two matrices containing {Q, L}
	 */
	public static final double[][][] eig_jacobi(final double[][] m, boolean prec) {
		int n = m.length;
		double[][] matrix;
		
		// init eigenvectors
		double[][] G;
		
		// check if need precond
		boolean diag = true;
		for(int i = 1 ; i < n ; i++)
			if(m[0][i] > m[0][0]/n)
				diag = false;
		
		if(!prec || diag)  {
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
			// tri precond
			double[][][] QT = tri_lancsos(m);
			matrix = QT[1];
			G = transi(QT[0]);
		}
		
		while(true) {
			for(int i = 0 ; i < n-1 ; i++) {
				// search largest off diag for i
				int j = i+1;
				for(int id = i+1 ; id < n ; id++) {
					if(abs(matrix[i][id]) > abs(matrix[i][j]))
						j = id;
				}
				if(abs(matrix[i][j]) > 1e-14)
					givensRoti(matrix, G, i, j);
			}
			
			// check if done
			boolean cont = false;
			for(int i = 0 ; i < n-1 && !cont; i++)
				for(int j = i+1 ; j < n && !cont; j++)
					if(abs(matrix[i][j]) > 1e-14)
						cont = true;
			if(!cont)
				break;
		}
		for(int i = 0 ; i < n ; i++)
			for(int j = i+1 ; j< n ; j++) {
				matrix[i][j] = 0;
				matrix[j][i] = 0;
			}
		return new double[][][] {transi(G), matrix};
	}

	
	/**
	 * Performs the eigen decomposition of a symmetric matrix:
	 * A = Q * L * Q'
	 * with Q orthonormal and L diagonal
	 * @param A input matrix
	 * @return an array of two matrices containing {Q, L}
	 */
	public static double[][][] eig_qr(double[][] A) {
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
	
	/**
	 * Performs the trigonalization of a symmetric matrix:
	 * A = Q * T * Q'
	 * with Q orthonormal and T tridiagonal
	 * @param m input matrix
	 * @return an array of two matrices containing {Q, L}
	 */
	public static double[][][] tri(final double[][] m){
		return tri_lancsos(m);
	}
	
	public static double[][][] tri_householder(final double[][]m) {
		int n = m.length;
		double[][] matrix;
		
		// first vector
		double alpha = 0;
		for(int j = 1 ; j < n ; j++)
			alpha += m[j][0]*m[j][0];
		alpha = -signum(m[1][0])*sqrt(alpha);
		double r = sqrt(0.5*(alpha*alpha - m[1][0]*alpha));
		double[] v = new double[n];
		v[0] = 0;
		v[1] = (m[1][0] - alpha)/(2*r);
		for(int j = 2 ; j < n ; j++)
			v[j] = m[j][0] / (2*r);

		// first projection matrix
		double[][] P = new double[n][n];
		for(int i = 0 ; i < n ; i++)
			for(int j = 0 ; j < n ; j++) {
				if(i == j)
					P[i][j] = 1;
				P[i][j] -= 2*v[i]*v[j];
			}
		// apply transform
		matrix = mul(P, mul(m, P));
		// store first transform
		double[][] Q = P;
		P = new double[n][n];

		// buffer for intermediate proj
		double[][] buffer = new double[n][n];
		
		// all other transforms
		for(int k = 1 ; k < n-2 ; k++) {
			//kth vector
			alpha = 0;
			for(int j = k+1 ; j < n ; j++)
				alpha += matrix[j][k]*matrix[j][k];
			alpha = -signum(matrix[k+1][k])*sqrt(alpha);
			r = sqrt(0.5*(alpha*alpha - matrix[k+1][k]*alpha));
			for(int j = 0 ; j <= k ; j++)
				v[j] = 0;
			v[k+1] = (matrix[k+1][k] - alpha)/(2*r);
			for(int j = k+2 ; j < n ; j++)
				v[j] = matrix[j][k] / (2*r);
			
			// kth proj matrix
			for(int i = 0 ; i < n ; i++){
				Arrays.fill(P[i], 0);
				P[i][i] = 1.;
			}
			for(int i = k; i < n ; i++)
				for(int j = i ; j < n ; j++) {
					P[i][j] -= 2*v[i]*v[j];
					P[j][i] = P[i][j];
				}

			// apply transform
			muli_alr(matrix, P, muli_blr(buffer,matrix, P, k), k);
			// save transform
			Q = mul(P, Q);
		}
		
		for(int i = 0 ; i < n ; i++)
			for(int j = i+2 ; j < n ; j++) {
				matrix[i][j] = 0;
				matrix[j][i] = 0;
			}
				
		return new double[][][]{Q, matrix};
	}
	
	private static double[][] muli_alr(double[][] C, final double[][] A, final double[][] B, int r) {
		int n, m, p;
		m = A.length;
		n = B[0].length;
		p = A[0].length;
		
		if(p != B.length) {
			throw new ArithmeticException("Matrix dimensions must agree.");
		}
		for(int i = 0 ; i < m ; i++) {
			for(int j = 0 ; j < n ; j++) {
				if(i < r)
					C[i][j] = B[i][j];
				else {
					double sum = 0;
					for(int k = 0 ; k < p ; k++) {
						sum += A[i][k]*B[k][j];
					}
					C[i][j] = sum;
				}
			}
		}
		return C;
	}
	private static double[][] muli_blr(double[][] C, final double[][] A, final double[][] B, int r) {
		int n, m, p;
		m = A.length;
		n = B[0].length;
		p = A[0].length;
		
		if(p != B.length) {
			throw new ArithmeticException("Matrix dimensions must agree.");
		}
		for(int i = 0 ; i < m ; i++) {
			for(int j = 0 ; j < n ; j++) {
				if(j < r)
					C[i][j] = A[i][j];
				else {
					double sum = 0;
					for(int k = 0 ; k < p ; k++) {
						sum += A[i][k]*B[k][j];
					}
					C[i][j] = sum;
				}
			}
		}
		return C;
	}
	
	public static double[][][] tri_lancsos(final double[][] m) {
		int n = m.length;
		double[] alpha = new double[n];
		double[] beta = new double[n];
		
		// init vector at random
		double[][] v = new double[n][n];
		for(int i = 0 ; i < n ; i++) {
			v[0][i] = 2*random()-1;
		}
		double Z = 1./n2(v[0]);
		VectorOperations.muli(v[0], v[0], Z);
		
		// project
		double[] w = new double[n];
		for(int i = 0 ; i < n ; i++)
			w[i] = dot(m[i], v[0]);
		
		// coeff
		alpha[0] = dot(w, v[0]);
		
		// orthogonalize
		addi(w, w, -alpha[0], v[0]);
		
		// norm 
		beta[1] = n2(w);
		
		// next
		VectorOperations.muli(v[1], w, 1./beta[1]);
		// orthogonalize
		addi(v[1], v[1], -dot(v[0], v[1]), v[0]);
		
		// all other
		for(int k = 1 ; k < n-1 ; k++) {
			for(int i = 0 ; i < n ; i++)
				w[i] = dot(m[i], v[k]);
			
			// coeff
			alpha[k] = dot(w, v[k]);
			
			// orthogonalize
			addi(w, w, -alpha[k], v[k]);
			addi(w, w, -beta[k], v[k-1]);
			// force ortho
			for(int j = 0 ; j < k+1 ; j++)
				addi(w, w, -dot(v[j], w), v[j]);
			
			// norm 
			beta[k+1] = n2(w);
			
			// next
			VectorOperations.muli(v[k+1], w, 1./beta[k+1]);
		}
		// final
		for(int i = 0 ; i < n ; i++)
			w[i] = dot(m[i], v[n-1]);
		
		// coeff
		alpha[n-1] = dot(w, v[n-1]);
		
		double[][] T = new double[n][n];
		for(int i = 0 ; i < n-1 ; i++) {
			T[i][i] = alpha[i];
			T[i][i+1] = beta[i+1];
			T[i+1][i] = beta[i+1];
		}
		T[n-1][n-1] = alpha[n-1];
		
		// transpose vectors to form basis
		transi(v);
		
		return new double[][][]{v, T};
	}
	
	public static double[][][] tri_givens(double[][] m, boolean prec) {
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
			for(int i = 0 ; i < n-2 ; i++) {
				// search largest off tridiag for i
				int j = i+2;
				for(int id = i+2 ; id < n ; id++) {
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
			for(int i = 0 ; i < n-2 ; i++)
				for(int j = i+2 ; j < n ; j++)
					if(abs(matrix[i][j]) > max)
						max = abs(matrix[i][j]);
			if(max < 1e-16)
				break;
		}
		for(int i = 0 ; i < n ; i++)
			for(int j = i+2 ; j< n ; j++) {
				matrix[i][j] = 0;
				matrix[j][i] = 0;
			}
		return new double[][][] {transi(G), matrix};
	}

}
