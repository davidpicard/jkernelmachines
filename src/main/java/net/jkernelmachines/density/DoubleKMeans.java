/*******************************************************************************
 * Copyright (c) 2016, David Picard.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************/
package net.jkernelmachines.density;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import net.jkernelmachines.util.DebugPrinter;
import net.jkernelmachines.util.algebra.VectorOperations;

/**
 * Very basic KMeans algorithm with a shifting codeword procedure to ensure no
 * empty cluster and balanced distortion
 * 
 * @author picard
 * 
 */
public class DoubleKMeans implements DensityFunction<double[]> {

	private static final long serialVersionUID = -376280133933635170L;

	int K;
	double[][] means;

	double shiftRatio = 20;

	DebugPrinter debug = new DebugPrinter();

	/**
	 * Constructor with number of clusters
	 * 
	 * @param k
	 *            number of clusters
	 */
	public DoubleKMeans(int k) {
		K = k;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.density.DensityFunction#train(java.lang.Object)
	 */
	@Override
	public void train(double[] e) {
		throw new UnsupportedOperationException(
				"Training on a single sample is not supported");
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.density.DensityFunction#train(java.util.List)
	 */
	@Override
	public void train(List<double[]> train) {
		int n = train.size();
		int dim = train.get(0).length;

		if (K > n) {
			throw new ArithmeticException("Too few data points: " + n + " < "
					+ K);
		}

		double[] w = new double[K];
		double[][] mu = new double[K][dim];
		Random rand = new Random();

		int t = 0;
		// init with k-means
		int c[] = new int[n];
		for (int i = 0; i < n; i++) {
			c[i] = rand.nextInt(K);
			VectorOperations.addi(mu[c[i]], mu[c[i]], 1, train.get(i));
			w[c[i]] += 1;
		}
		for (int g = 0; g < K; g++) {
			if (w[g] > 0) {
				VectorOperations.muli(mu[g], mu[g], 1. / w[g]);
			} else {
				Arrays.fill(mu[g], 0);
			}
		}

		for (; t < 10000; t++) {
			boolean cont = false;
			// E
			for (int i = 0; i < n; i++) {
				double[] x = train.get(i);
				double dmin = Double.POSITIVE_INFINITY;
				int cmin = -1;
				for (int g = 0; g < K; g++) {
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

			// M
			for (int g = 0; g < K; g++) {
				Arrays.fill(mu[g], 0);
				w[g] = 0;
			}
			for (int i = 0; i < n; i++) {
				VectorOperations.addi(mu[c[i]], mu[c[i]], 1, train.get(i));
				w[c[i]] += 1;
			}
			for (int g = 0; g < K; g++) {
				if (w[g] > 0) {
					VectorOperations.muli(mu[g], mu[g], 1. / w[g]);
				} else {
					Arrays.fill(mu[g], 0);
				}
			}

			if (!cont) {
				// try codeword shifting
				double[] dist = new double[K];
				double dtot = 0;
				for (int i = 0; i < n; i++) {
					double[] x = train.get(i);
					double d = VectorOperations.d2p2(x, mu[c[i]]);
					dist[c[i]] += d;
					dtot += d;
				}
				debug.println(3, "d: " + Arrays.toString(dist));
				debug.println(2, "total dist: " + dtot);
				double dmin = Double.POSITIVE_INFINITY, dmax = -1;
				int imin = -1, imax = -1;
				for (int g = 0; g < K; g++) {
					if (dist[g] < dmin) {
						dmin = dist[g];
						imin = g;
					}
					if (dist[g] > dmax) {
						dmax = dist[g];
						imax = g;
					}
				}
				debug.println(3, "dmin: " + dmin + "\tdmax: " + dmax);
				if (dmin == 0 || dmax / dmin > shiftRatio) {
					// shift
					int comp = rand.nextInt(dim);
					mu[imin] = Arrays.copyOf(mu[imax], dim);
					double no = VectorOperations.n2(mu[imax]);
//					for(int comp = 0 ; comp < dim ; comp++) {
						mu[imin][comp] += 1e-6*no;
						mu[imax][comp] -= 1e-6*no;
//					}
					debug.println(2, "shifting done");
				} else {
					break;
				}
			}
		}

		// save means
		means = mu;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.density.DensityFunction#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(double[] e) {
		double dmin = Double.POSITIVE_INFINITY;
		int index = -1;
		for (int g = 0; g < K; g++) {
			double d = VectorOperations.d2p2(e, means[g]);
			if (d < dmin) {
				dmin = d;
				index = g;
			}
		}
		return index;
	}

	/**
	 * Return an array containing the squared distances to each clusters
	 * 
	 * @param e
	 *            the sample to evaluate
	 * @return the array of distances
	 */
	public double[] distanceToMean(double[] e) {
		double[] d = new double[K];
		for (int g = 0; g < K; g++) {
			d[g] = VectorOperations.d2p2(e, means[g]);
		}
		return d;
	}
}
