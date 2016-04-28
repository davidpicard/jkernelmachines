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
package net.jkernelmachines.classifier;

import static java.lang.Math.min;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import net.jkernelmachines.density.DoubleKMeans;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.DebugPrinter;
import net.jkernelmachines.util.algebra.MatrixVectorOperations;
import net.jkernelmachines.util.algebra.VectorOperations;

/**
 * <p>
 * Locally Linear SVM, as described in: Locally Linear Support Vector Machines
 * L'ubor Ladicky, Philip H.S. Torr Procedings of the 28th ICML, Bellevue, WA,
 * USA, 2011.
 * </p>
 * 
 * @author picard
 *
 */
public class DoubleLLSVM implements Classifier<double[]> {

	DoubleKMeans km;
	double[][] W;
	double[] b;

	int K = 32;
	int E = 10;
	long t0 = 100;
	int skip = 10;
	double C = 1.0;
	int nn = 2;

	DebugPrinter debug = new DebugPrinter();

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#train(java.util.List)
	 */
	@Override
	public void train(List<TrainingSample<double[]>> l) {

		nn = min(nn, K);

		// train gmm
		List<double[]> list = new ArrayList<>(l.size());
		for (TrainingSample<double[]> t : l) {
			list.add(t.sample);
		}
		km = new DoubleKMeans(K);
		km.train(list);
		debug.println(1, "KM trained");

		// compute likelihoods
		list.clear();
		List<Integer> index = new LinkedList<>();
		for (int i = 0; i < l.size(); i++) {
			double[] d = likelihood(l.get(i).sample);
			list.add(d);
			index.add(i);
		}
		debug.println(1, "likelihood computed");

		// sgd on parameters
		W = new double[K][l.get(0).sample.length];
		b = new double[K];

		int count = skip;
		long t = t0;
		double lambda = 1. / (C * l.size());
		for (int e = 0; e < E; e++) {
			Collections.shuffle(index);
			for (int i : index) {
				double[] g = list.get(i);
				double[] x = l.get(i).sample;
				int y = l.get(i).label;

				// W
				double v = VectorOperations.dot(b, g);
				v += VectorOperations.dot(g, MatrixVectorOperations.rMul(W, x));
				if (1 - y * v > 0) {
					double[][] grad = MatrixVectorOperations.outer(g, x);
					for (int m = 0; m < g.length; m++) {
						for (int n = 0; n < x.length; n++) {
							W[m][n] += 1. / (lambda * t) * y * grad[m][n];
						}
					}
				}

				// b
				VectorOperations.addi(b, b, 1. / (lambda * t) * y, g);

				if (--count < 0) {
					count = skip;
					for (int m = 0; m < g.length; m++) {
						for (int n = 0; n < x.length; n++) {
							W[m][n] *= 1 - skip / t;
						}
					}
				}

				t++;
			}

			// last regularization
			if (count > 0) {
				for (int m = 0; m < W.length; m++) {
					for (int n = 0; n < W[m].length; n++) {
						W[m][n] *= 1 - (skip - count) / t;
					}
				}
			}

			debug.println(1, "epoch " + e + " finished");
		}

		debug.println(2, "W: " + Arrays.deepToString(W));
		debug.println(2, "b: " + Arrays.toString(b));

	}

	private double[] likelihood(double[] e) {
		double[] d = km.distanceToMean(e);
		double[] ds = Arrays.copyOf(d, d.length);
		Arrays.sort(ds);
		double threshold = ds[nn - 1];
		double sum = 0;
		for (int g = 0; g < K; g++) {
			if (d[g] <= threshold) {
				d[g] = 1 / (1 + d[g]);
			} else {
				d[g] = 0;
			}
			sum += d[g];
		}
		// norm to unit sum
		VectorOperations.mul(d, 1. / sum);
		return d;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.classifier.Classifier#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(double[] e) {

		double[] d = likelihood(e);
		double out = VectorOperations.dot(d, MatrixVectorOperations.rMul(W, e));
		out += VectorOperations.dot(d, b);

		return out;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#copy()
	 */
	@Override
	public Classifier<double[]> copy() throws CloneNotSupportedException {
		return (DoubleLLSVM) this.clone();
	}

	/**
	 * return the number of anchor points
	 * 
	 * @return the number of anchor points
	 */
	public int getK() {
		return K;
	}

	/**
	 * Sets the number of anchor points
	 * 
	 * @param k
	 *            the number of anchor points
	 */
	public void setK(int k) {
		K = k;
	}

	/**
	 * Returns the number of epochs for training
	 * 
	 * @return the number of epochs
	 */
	public int getE() {
		return E;
	}

	/**
	 * Sets the number of epochs for training
	 * 
	 * @param e
	 *            the number of epochs
	 */
	public void setE(int e) {
		E = e;
	}

	/**
	 * Returns the hyperparameter C for the hinge loss tradeoff
	 * 
	 * @return C C
	 */
	public double getC() {
		return C;
	}

	/**
	 * Sets the hyperparameter C for the hinge loss tradeoff
	 * 
	 * @param c C
	 */
	public void setC(double c) {
		C = c;
	}

	/**
	 * Returns the number of anchor points taken into account by the model
	 * 
	 * @return the number of anchor points
	 */
	public int getNn() {
		return nn;
	}

	/**
	 * Sets the number of anchor points taken into account by the model
	 * 
	 * @param nn
	 *            the number of anchor opints
	 */
	public void setNn(int nn) {
		this.nn = nn;
	}

	/**
	 * Return the model hyperplanes
	 * 
	 * @return W
	 */
	public double[][] getW() {
		return W;
	}

	/**
	 * Return the model biases
	 * 
	 * @return b
	 */
	public double[] getB() {
		return b;
	}

}
