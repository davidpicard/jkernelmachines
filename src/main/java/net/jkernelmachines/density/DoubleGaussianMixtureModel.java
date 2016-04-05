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
package net.jkernelmachines.density;

import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.sqrt;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import net.jkernelmachines.util.DebugPrinter;
import net.jkernelmachines.util.algebra.MatrixOperations;
import net.jkernelmachines.util.algebra.MatrixVectorOperations;
import net.jkernelmachines.util.algebra.VectorOperations;

/**
 * <p>
 * Gaussian Mixture Model for estimating the density on arrays of double.
 * </p>
 * <p>
 * The initialization of the Gaussian centers is performed by k-means, then the
 * EM algorithm runs until the centers stabilize.
 * </p>
 * 
 * @author picard
 * 
 */
public class DoubleGaussianMixtureModel implements DensityFunction<double[]> {

	private static final long serialVersionUID = -6989529384513214743L;

	int k;
	double[] w;
	double[][] mu;
	double[][][] sigma;

	List<double[]> train;

	DebugPrinter debug = new DebugPrinter();

	public DoubleGaussianMixtureModel(int nb_gaussian) {
		k = nb_gaussian;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.density.DensityFunction#train(java.lang.Object)
	 */
	@Override
	public void train(double[] e) {
		// TODO Auto-generated method stub
		if (train == null) {
			train = new ArrayList<double[]>();
		}
		train.add(e);
		train(train);

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.density.DensityFunction#train(java.util.List)
	 */
	@Override
	public void train(List<double[]> e) {
		train = new ArrayList<double[]>();
		train.addAll(e);

		int n = train.size();
		int dim = train.get(0).length;

		w = new double[k];
		mu = new double[k][dim];
		sigma = new double[k][dim][dim];

		double[] oldmu = new double[dim];

		// init
		double[][] like = new double[n][k];

		Random rand = new Random();

		int t = 0;
		// init with k-means
		int c[] = new int[n];
		for (int i = 0; i < n; i++) {
			c[i] = rand.nextInt(k);
			VectorOperations.addi(mu[c[i]], mu[c[i]], 1, train.get(i));
			w[c[i]] += 1;
		}
		for (int g = 0; g < k; g++) {
			if(w[g] > 0) {
				VectorOperations.muli(mu[g], mu[g], 1. / w[g]);
			}
			else {
				Arrays.fill(mu[g], 0);
			}
		}

		for (; t < 1000; t++) {
			boolean cont = false;
			// E
			for (int i = 0; i < n; i++) {
				double[] x = train.get(i);
				double dmin = Double.POSITIVE_INFINITY;
				int cmin = -1;
				for (int g = 0; g < k; g++) {
					double d = VectorOperations.d2p2(x, mu[g]);
					if (d < dmin) {
						cmin = g;
						dmin = d;
					}
				}
				if (cmin != c[i])
					cont = true;
				c[i] = cmin;
			}

			if (!cont)
				break;

			// M
			for (int g = 0; g < k; g++) {
				Arrays.fill(mu[g], 0);
				w[g] = 0;
			}
			for (int i = 0; i < n; i++) {
				VectorOperations.addi(mu[c[i]], mu[c[i]], 1, train.get(i));
				w[c[i]] += 1;
			}
			for (int g = 0; g < k; g++) {
				if(w[g] > 0) {
					VectorOperations.muli(mu[g], mu[g], 1. / w[g]);
				}
				else {
					Arrays.fill(mu[g], 0);
				}
			}
		}

		debug.println(3, "init k-means :");
		debug.println(3, " t = " + t);
		for (int i = 0; i < k; i++)
			debug.println(3, " mu" + i + " = " + Arrays.toString(mu[i]));

		// diag cov
		for (int g = 0; g < k; g++) {
			w[g] /= n;
			for (int d = 0; d < dim; d++) {
				sigma[g][d][d] = 1.0;
			}
		}
		for (t = 0; t < 10000; t++) {
			// E
			for (int x = 0; x < n; x++) {
				double[] xt = train.get(x);
				double sum = 0;
				for (int g = 0; g < k; g++) {
					double[] xtc = VectorOperations.add(xt, -1, mu[g]);
					double[] o = MatrixVectorOperations.rMul(sigma[g], xtc);
					like[x][g] = (w[g])
							* exp((-0.5 * VectorOperations.dot(xtc, o)));
					if (like[x][g] < 1e-15)
						like[x][g] = 0;
					sum += like[x][g];
				}
				if (abs(sum) > 1e-15)
					VectorOperations.muli(like[x], like[x], 1. / sum);
			}
			// M
			for (int g = 0; g < k; g++) {
				w[g] = 0;
				Arrays.fill(mu[g], 0);
				for (int d = 0; d < dim; d++) {
					Arrays.fill(sigma[g][d], 0);
				}
			}
			// weights and means
			for (int x = 0; x < n; x++) {
				double[] xt = train.get(x);
				for (int g = 0; g < k; g++) {
					w[g] += like[x][g];
					VectorOperations.addi(mu[g], mu[g], like[x][g], xt);
				}
			}
			for (int g = 0; g < k; g++) {
				VectorOperations.muli(mu[g], mu[g], 1. / w[g]);
			}
			// covariances
			for (int x = 0; x < n; x++) {
				double[] xt = train.get(x);
				for (int g = 0; g < k; g++) {
					double[] xtc = VectorOperations.add(xt, -1, mu[g]);
					VectorOperations.muli(xtc, xtc, sqrt(like[x][g]));
					MatrixVectorOperations.addXXTrans(sigma[g], xtc);
				}
			}
			for (int g = 0; g < k; g++) {
				for (int d = 0; d < dim; d++) {
					VectorOperations.muli(sigma[g][d], sigma[g][d],
							1. / (2 * w[g]));
				}
				sigma[g] = MatrixOperations.inv(sigma[g]);
			}
			// normalize weights
			VectorOperations.muli(w, w, 1. / n);

			// if no improvements, quit
			if (VectorOperations.d2p2(mu[0], oldmu)
					/ VectorOperations.n2p2(oldmu) < 1e-12)
				break;
			oldmu = Arrays.copyOf(mu[0], dim);

		}
		debug.println(3, "t = " + t);
		debug.println(3, "pop= " + Arrays.toString(w));

		for (int i = 0; i < k; i++)
			debug.println(3, "mu" + i + " = " + Arrays.toString(mu[i]));

		for (int i = 0; i < k; i++) {
			debug.println(3,
					"sigma" + i + " = " + Arrays.deepToString(sigma[i]));
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.density.DensityFunction#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(double[] e) {
		double sum = 0;
		for (int g = 0; g < k; g++) {
			double[] xtc = VectorOperations.add(e, -1, mu[g]);
			double[] o = MatrixVectorOperations.rMul(sigma[g], xtc);
			sum += w[g] * exp(-0.5 * VectorOperations.dot(xtc, o));

		}
		return sum;
	}
	
	/**
	 * Return a vector containing the likelihood to each Gaussian component
	 * @param e the sample to evaluate
	 * @return the vector of likelihood
	 */
	public double[] likelihood(double[] e) {
		double[] l = new double[k];
		
		for (int g = 0; g < k; g++) {
			double[] xtc = VectorOperations.add(e, -1, mu[g]);
			double[] o = MatrixVectorOperations.rMul(sigma[g], xtc);
			l[g] = w[g] * exp(-0.5 * VectorOperations.dot(xtc, o));

		}		
		
		return l;
	}

	/**
	 * Get the number of components in the mixture
	 * 
	 * @return the number of Gaussian
	 */
	public int getK() {
		return k;
	}

	/**
	 * Sets the number of components in the mixture
	 * 
	 * @param k
	 *            the number of Gaussian
	 */
	public void setK(int k) {
		this.k = k;
	}

	/**
	 * Gets the weights associated with each component of the mixture
	 * 
	 * @return an array containing the weights of each Gaussian
	 */
	public double[] getW() {
		return w;
	}

	/**
	 * Sets the weights associated with each Component of the mixture
	 * 
	 * @param w
	 *            an array containing the weights of each Gaussian
	 */
	public void setW(double[] w) {
		this.w = w;
	}

	/**
	 * Gets the centers of each component of the mixture
	 * 
	 * @return an array of double arrays, each being the center of a Gaussian
	 */
	public double[][] getMu() {
		return mu;
	}

	/**
	 * Sets the centers of each component of the mixture
	 * 
	 * @param mu
	 *            an array of of double arrays, each being the center of the
	 *            coresponding Gaussian
	 */
	public void setMu(double[][] mu) {
		this.mu = mu;
	}

	/**
	 * Gets the inverse covariance matrices of each component of the mixture
	 * 
	 * @return an array of double[][] arrays, each being the inverse covariance
	 *         matrix of associated Gaussian
	 */
	public double[][][] getSigma() {
		return sigma;
	}

	/**
	 * Sets the inverse covariance matrices of each component of the mixture
	 * 
	 * @param sigma
	 *            an array of double[][], each being the inverse covariance
	 *            matrix of associated Gaussian
	 */
	public void setSigma(double[][][] sigma) {
		this.sigma = sigma;
	}

}
