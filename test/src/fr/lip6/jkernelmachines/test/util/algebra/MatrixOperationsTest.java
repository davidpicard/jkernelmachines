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
package fr.lip6.jkernelmachines.test.util.algebra;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import fr.lip6.jkernelmachines.util.algebra.MatrixOperations;

/**
 * @author picard
 * 
 */
public class MatrixOperationsTest {

	/**
	 * Test method for
	 * {@link fr.lip6.jkernelmachines.util.algebra.MatrixOperations#isSquare(double[][])}
	 * .
	 */
	@Test
	public final void testIsSquare() {
		double[][] A = new double[2][2];
		double[][] B = new double[3][2];

		assertEquals(true, MatrixOperations.isSquare(A));
		assertEquals(false, MatrixOperations.isSquare(B));
	}

	/**
	 * Test method for
	 * {@link fr.lip6.jkernelmachines.util.algebra.MatrixOperations#isSymmetric(double[][])}
	 * .
	 */
	@Test
	public final void testIsSymmetric() {
		double[][] A = { { 1, 2 }, { 2, 1 } };

		assertEquals(true, MatrixOperations.isSymmetric(A));

		A[0][1] = 0;
		assertEquals(false, MatrixOperations.isSymmetric(A));
	}

	/**
	 * Test method for
	 * {@link fr.lip6.jkernelmachines.util.algebra.MatrixOperations#trans(double[][])}
	 * .
	 */
	@Test
	public final void testTrans() {
		double[][] A = { { 1, 2 }, { 3, 4 } };

		double[][] B = MatrixOperations.trans(A);
		assertEquals(1, B[0][0], 1e-15);
		assertEquals(3, B[0][1], 1e-15);
		assertEquals(2, B[1][0], 1e-15);
		assertEquals(4, B[1][1], 1e-15);
	}

	/**
	 * Test method for
	 * {@link fr.lip6.jkernelmachines.util.algebra.MatrixOperations#transi(double[][])}
	 * .
	 */
	@Test
	public final void testTransi() {
		double[][] A = { { 1, 2 }, { 3, 4 } };

		MatrixOperations.transi(A);
		assertEquals(1, A[0][0], 1e-15);
		assertEquals(3, A[0][1], 1e-15);
		assertEquals(2, A[1][0], 1e-15);
		assertEquals(4, A[1][1], 1e-15);
	}

	/**
	 * Test method for
	 * {@link fr.lip6.jkernelmachines.util.algebra.MatrixOperations#transi(double[][], double[][])}
	 * .
	 */
	@Test
	public final void testTransi2() {
		double[][] A = { { 1, 2 }, { 3, 4 } };
		double[][] C = { { 0, 0 }, { 0, 0 } };

		MatrixOperations.transi(C, A);
		assertEquals(1, C[0][0], 1e-15);
		assertEquals(3, C[0][1], 1e-15);
		assertEquals(2, C[1][0], 1e-15);
		assertEquals(4, C[1][1], 1e-15);
	}

	/**
	 * Test method for
	 * {@link fr.lip6.jkernelmachines.util.algebra.MatrixOperations#mul(double[][], double[][])}
	 * .
	 */
	@Test
	public final void testMul() {
		double[][] A = { { 2, 1 }, { 3, 0 }, { 4, 0 } };
		double[][] B = { { 0, 1, 0 }, { 1000, 100, 10 } };

		double[][] C = MatrixOperations.mul(A, B);

		assertEquals(1000, C[0][0], 1e-15);
		assertEquals(102, C[0][1], 1e-15);
		assertEquals(10, C[0][2], 1e-15);
		assertEquals(0, C[1][0], 1e-15);
		assertEquals(3, C[1][1], 1e-15);
		assertEquals(0, C[1][2], 1e-15);
		assertEquals(0, C[2][0], 1e-15);
		assertEquals(4, C[2][1], 1e-15);
		assertEquals(0, C[2][2], 1e-15);
	}

	/**
	 * Test method for
	 * {@link fr.lip6.jkernelmachines.util.algebra.MatrixOperations#muli(double[][], double[][], double[][])}
	 * .
	 */
	@Test
	public final void testMuli() {
		double[][] A = { { 2, 1 }, { 3, 0 }, { 4, 0 } };
		double[][] B = { { 0, 1, 0 }, { 1000, 100, 10 } };

		double[][] C = new double[3][3];

		MatrixOperations.muli(C, A, B);

		assertEquals(1000, C[0][0], 1e-15);
		assertEquals(102, C[0][1], 1e-15);
		assertEquals(10, C[0][2], 1e-15);
		assertEquals(0, C[1][0], 1e-15);
		assertEquals(3, C[1][1], 1e-15);
		assertEquals(0, C[1][2], 1e-15);
		assertEquals(0, C[2][0], 1e-15);
		assertEquals(4, C[2][1], 1e-15);
		assertEquals(0, C[2][2], 1e-15);
	}

	/**
	 * Test method for
	 * {@link fr.lip6.jkernelmachines.util.algebra.MatrixOperations#transMul(double[][], double[][])}
	 * .
	 */
	@Test
	public final void testTransMul() {
		double[][] A = { { 2, 1 }, { 3, 0 } };
		double[][] B = { { 0, 1 }, { 1000, 100 } };

		double[][] C = MatrixOperations.mul(MatrixOperations.trans(A), B);

		double[][] Cprime = MatrixOperations.transMul(A, B);

		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				assertEquals(C[i][j], Cprime[i][j], 1e-15);
			}
		}
	}

	/**
	 * Test method for
	 * {@link fr.lip6.jkernelmachines.util.algebra.MatrixOperations#transMuli(double[][], double[][], double[][])}
	 * .
	 */
	@Test
	public final void testTransMuli() {
		double[][] A = { { 2, 1 }, { 3, 0 } };
		double[][] B = { { 0, 1 }, { 1000, 100 } };

		double[][] C = MatrixOperations.mul(MatrixOperations.trans(A), B);

		double[][] Cprime = new double[2][2];
		MatrixOperations.transMuli(Cprime, A, B);

		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				assertEquals(C[i][j], Cprime[i][j], 1e-15);
			}
		}
	}

	/**
	 * Test method for
	 * {@link fr.lip6.jkernelmachines.util.algebra.MatrixOperations#qr(double[][])}
	 * .
	 */
	@Test
	public final void testQr() {
		double[][] A = { { 12, 6, -4 }, { -51, 167, 24 }, { 4, -68, -41 } };

		double[][][] QR = MatrixOperations.qr(A);

		// is Q orthogonal?
		double[][] QtQ = MatrixOperations.transMul(QR[0], QR[0]);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				if (i == j) {
					assertEquals(1.0, QtQ[i][j], 1e-15);
				} else {
					assertEquals(0, QtQ[i][j], 1e-15);
				}
			}
		}

		// is R upper triangle
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < i; j++) {
				assertEquals(0.0, QR[1][i][j], 1e-15);
			}
		}

		// Does Q*R reconstruct A
		double[][] C = MatrixOperations.mul(QR[0], QR[1]);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				assertEquals(A[i][j], C[i][j], 1e-10);
			}
		}
	}

	/**
	 * Test method for
	 * {@link fr.lip6.jkernelmachines.util.algebra.MatrixOperations#qri_gramschmidt(double[][], double[][], double[][])}
	 * .
	 */
	@Test
	public final void testQri() {
		double[][] A = { { 12, 6, -4 }, { -51, 167, 24 }, { 4, -68, -41 } };

		double[][] Q = new double[A.length][A[0].length];
		double[][] R = new double[A.length][A[0].length];

		MatrixOperations.qri(Q, R, A);

		// is Q orthogonal?
		double[][] QtQ = MatrixOperations.transMul(Q, Q);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				if (i == j) {
					assertEquals(1.0, QtQ[i][j], 1e-15);
				} else {
					assertEquals(0, QtQ[i][j], 1e-15);
				}
			}
		}

		// is R upper triangle
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < i; j++) {
				assertEquals(0.0, R[i][j], 1e-15);
			}
		}

		// Does Q*R reconstruct A
		double[][] C = MatrixOperations.mul(Q, R);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				assertEquals(A[i][j], C[i][j], 1e-10);
			}
		}
	}

	/**
	 * Test method for
	 * {@link fr.lip6.jkernelmachines.util.algebra.MatrixOperations#eig(double[][])}
	 * .
	 */
	@Test
	public final void testEig() {
		double[][] A = { { 4, 3, 1 }, { 3, -3, -2 }, { 1, -2, 2 } };

		double[][][] eig = MatrixOperations.eig(A);

		// is U orthogonal?
		double[][] UtU = MatrixOperations.transMul(eig[0], eig[0]);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				if (i == j) {
					assertEquals(1.0, UtU[i][j], 1e-10);
				} else {
					assertEquals(0, UtU[i][j], 1e-10);
				}
			}
		}

		// is L diagonal
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				if (i != j) {
					assertEquals(0.0, eig[1][i][j], 1e-10);
				}
			}
		}

		// Does Q*L*Q' reconstruct A
		double[][] C = MatrixOperations.mul(eig[0],
				MatrixOperations.mul(eig[1], MatrixOperations.trans(eig[0])));
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				assertEquals(A[i][j], C[i][j], 1e-10);
			}
		}

		// test larger matrix
		int n = 256;
		double[][] X = new double[n][n];
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
				X[i][j] = Math.random() * 2 - 0.5;

		double[][] G = MatrixOperations.transMul(X, X);
		double[][][] ei = MatrixOperations.eig(G);
		// is U orthogonal?
		UtU = MatrixOperations.transMul(ei[0], ei[0]);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (i == j) {
					assertEquals(1.0, UtU[i][j], 1e-10);
				} else {
					assertEquals(0, UtU[i][j], 1e-10);
				}
			}
		}
		// is L diagonal
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (i != j) {
					assertEquals(0.0, ei[1][i][j], 1e-10);
				} else {
					assertTrue((ei[1][i][i] + 1e-10) >= 0);
				}
			}
		}
		double[][] rec = MatrixOperations.mul(ei[0],
				MatrixOperations.mul(ei[1], MatrixOperations.trans(ei[0])));
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				assertEquals(G[i][j], rec[i][j], 1e-10);
			}
		}

	}

	/**
	 * Test method for {@link
	 * fr.lip6.jkernelmachines.util.algebra.MatrixOperations#inv(final
	 * double[][])}.
	 */
	@Test
	public final void testInv() {
		double[][] A = { { 4, 0, 0 }, { 0, 3, 0 }, { 0, 0, 2 } };

		double[][] Ainv = MatrixOperations.inv(A);

		// A*(invA) = I
		double[][] I = MatrixOperations.mul(A, Ainv);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				if (i == j) {
					assertEquals(1.0, I[i][j], 1e-10);
				} else {
					assertEquals(0.0, I[i][j], 1e-10);
				}
			}
		}

		// test larger matrix
		int n = 128;
		double[][] X = new double[n][n];
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
				X[i][j] = Math.random() * 2 - 1.0;

		double[][] G = MatrixOperations.transMul(X, X);
		for(int i = 0 ; i < n ; i++) {
			G[i][i] += 1.0;
		}
		// A*(invA) = I
		double[][] Ginv = MatrixOperations.inv(G);

		I = MatrixOperations.mul(G, Ginv);
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (i == j) {
					assertEquals(1.0, I[i][j], 1e-10);
				} else {
					assertEquals(0.0, I[i][j], 1e-10);
				}
			}
		}
	}

	/**
	 * Test method for
	 * {@link fr.lip6.jkernelmachines.util.algebra.MatrixOperations#tri(double[][])}
	 * .
	 */
	@Test
	public final void testTri() {

		// test larger matrix
		int n = 128;
		double[][] X = new double[n][n];
		for (int i = 0; i < n; i++)
			for (int j = 0; j < n; j++)
				X[i][j] = Math.random() * 2 - 0.5;

		double[][] G = MatrixOperations.transMul(X, X);
		double[][][] qt = MatrixOperations.tri(G);
		double[][] rec = MatrixOperations.mul(qt[0],
				MatrixOperations.mul(qt[1], MatrixOperations.trans(qt[0])));
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				assertEquals(G[i][j], rec[i][j], 1e-10);
			}
		}

	}

}
