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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import net.jkernelmachines.kernel.Kernel;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.DebugPrinter;

/**
 * <p>
 * Kernel SVM classifier implementing LaSVM-I algorithm
 * </p>
 * 
 * <p>
 * <b>Nonconvex Online Support Vector Machines</b>
 * Seyda Ertekin, Leon Bottou and C. Lee Giles
 * IEEE Transaction on Pattern Analysis and Machine Intelligence, 33(2):368–381,
 * Feb 2011
 * </p>
 * 
 * @author picard
 * 
 */
public class LaSVMI<T> implements KernelSVM<T> {

	Kernel<T> kernel;

	List<TrainingSample<T>> train;
	double[] alpha;
	double b = 0;

	boolean[] keset;
	double[] gset;
	double[] A;
	double[] B;

	double C = 1.0;
	double tau = 1e-15;
	long E = 5;
	int m = 100; // max non-sv in expansion
	double s = -1.0; // ramp loss param

	DebugPrinter debug = new DebugPrinter();
	boolean cache = true;
	double[][] kmatrix;

	/**
	 * Default constructor provideing the kernel
	 * 
	 * @param k kernel
	 */
	public LaSVMI(Kernel<T> k) {
		kernel = k;
	}

	/**
	 * Incremental train adding a single sample
	 * (performs a full retrain on the whole list a samples)
	 * @param t sample
	 */
	public void train(TrainingSample<T> t) {
		if (train == null)
			train = new ArrayList<TrainingSample<T>>();
		train.add(t);
		train(train);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.classifier.Classifier#train(java.util.List)
	 */
	@Override
	public void train(List<TrainingSample<T>> l) {

		train = new ArrayList<TrainingSample<T>>();
		train.addAll(l);

		// 1) initialization
		alpha = new double[train.size()];
		Arrays.fill(alpha, 0.);
		keset = new boolean[train.size()];
		Arrays.fill(keset, false);
		gset = new double[train.size()];
		Arrays.fill(gset, 0.);
		A = new double[train.size()];
		B = new double[train.size()];

		// max number of non SV in expansion
		m = Math.min(1 + train.size() / 100, 100);

		if (cache)
			kmatrix = kernel.getKernelMatrix(train);

		// 2) online iterations
		for (int e = 0; e < E; e++)
			for (int i = 0; i < train.size(); i++) {
				// early filtering with ramp loss
				double z = 0;
				if (cache)
					for (int n = 0; n < train.size(); n++) {
						if (keset[n])
							z += alpha[n] * kmatrix[i][n];
					}
				else {
					T xi = train.get(i).sample;
					for (int n = 0; n < train.size(); n++) {
						if (keset[n])
							z += alpha[n]
									* kernel.valueOf(train.get(n).sample, xi);
					}
				}
				z = train.get(i).label * z;
				if (z > 1 || z < s)
					continue;

				// compute target gap G
				double G = computeGapTarget();

				// threshold
				double threshold = Math.max(C, G);

				// run process(xi)
				process(i);

				// reprocess while gap > G and something to optimize
				int max = 1000;
				while (computeGap() > threshold && max > 0) {
					if (!reprocess())
						break;
					max--;
				}

				// periodically run clean
				if (i % (10 * m) == 0)
					clean();
			}
		clean();


	}

	private void clean() {
		// number of non-SV
		int nz = 0;
		double[] gnsv = new double[train.size()];
		for (int n = 0; n < train.size(); n++) {
			if (keset[n] && alpha[n] == 0) {
				nz++;
				gnsv[n] = Math.abs(gset[n]);
			}
		}

		// too many zeros, cleaning
		if (nz > m) {
			Arrays.sort(gnsv);
			int i = 0;
			while ((i + m) < gnsv.length && gnsv[i] == 0)
				i++;
			if (i + m < gnsv.length) {
				double gthreshold = gnsv[i + m];
				for (int n = 0; n < train.size(); n++) {
					if (keset[n] && alpha[n] == 0
							&& Math.abs(gset[n]) > gthreshold) {
						alpha[n] = 0.;
						keset[n] = false;
					}
				}
			}
		}
	}

	private boolean reprocess() {

		debug.println(4, "- reprocess()");

		int i = -1, j = -1;
		double min = Double.POSITIVE_INFINITY, max = Double.NEGATIVE_INFINITY;

		// finding argmin and argmax
		for (int n = 0; n < train.size(); n++) {
			if (keset[n]) {
				if (gset[n] < min && alpha[n] > A[n]) {
					i = n;
					min = gset[n];
				}
				if (gset[n] > max && alpha[n] < B[n]) {
					j = n;
					max = gset[n];
				}
			}
		}
		// no extrema
		if (i == -1 || j == -1)
			return false;

		// 2. violating pair?
		if ((gset[i] - gset[j]) > tau) {

			double g = 0;
			int t = 0;

			// 3. which one
			if (gset[i] + gset[j] < 0) {
				g = gset[i];
				t = i;
			} else {
				g = gset[j];
				t = j;
			}
			T xt = train.get(t).sample;
			// 4. step size
			double lambda = 0;
			if (g < 0) {
				if (cache)
					lambda = Math.max(A[t] - alpha[t], g / kmatrix[t][t]);
				else
					lambda = Math.max(A[t] - alpha[t],
							g / kernel.valueOf(xt, xt));
			} else {
				if (cache)
					lambda = Math.min(B[t] - alpha[t], g / kmatrix[t][t]);
				else
					lambda = Math.min(B[t] - alpha[t],
							g / kernel.valueOf(xt, xt));
			}
			// 5. update kernel expansion set
			alpha[t] += lambda;
			for (int n = 0; n < train.size(); n++) {
				if (keset[n])
					if (cache)
						gset[n] -= lambda * kmatrix[t][n];
					else
						gset[n] -= lambda
								* kernel.valueOf(xt, train.get(n).sample);
			}

			return true;
		}
		return false;

	}

	private void process(int i) {

		debug.println(4, "+ process()");

		TrainingSample<T> xi = train.get(i);

		// 1.
		// set new alpha
		alpha[i] = 0;
		keset[i] = true;
		// set new gradient double g
		double gi = xi.label;
		for (int n = 0; n < train.size(); n++) {
			if (keset[n]) {
				if (cache)
					gi -= alpha[n]
							* kernel.valueOf(xi.sample, train.get(n).sample);
				else
					gi -= alpha[n] * kmatrix[i][n];
			}
		}
		gset[i] = gi;
		A[i] = Math.min(0, C * xi.label);
		B[i] = Math.max(0, C * xi.label);

		// 2. step size
		double lambda = 0;
		if (gi < 0) { // max(Ai, gi/Kii)
			if (cache)
				lambda = Math.max(A[i] - alpha[i], gi / kmatrix[i][i]);
			else
				lambda = Math.max(A[i] - alpha[i],
						gi / kernel.valueOf(xi.sample, xi.sample));
		} else { // max(Bi, gi/Kii)
			if (cache)
				lambda = Math.min(B[i] - alpha[i], gi / kmatrix[i][i]);
			else
				lambda = Math.min(B[i] - alpha[i],
						gi / kernel.valueOf(xi.sample, xi.sample));
		}

		// 3. insertion
		alpha[i] = alpha[i] + lambda;
		if (cache)
			for (int n = 0; n < train.size(); n++) {
				if (keset[n]) {
					gset[n] -= lambda * kmatrix[i][n];
				}
			}
		else
			for (int n = 0; n < train.size(); n++) {
				if (keset[n]) {
					gset[n] -= lambda
							* kernel.valueOf(xi.sample, train.get(n).sample);
				}
			}

	}

	private double computeGap() {
		double G = 0;

		for (int n = 0; n < train.size(); n++) {
			if (keset[n]) {
				// G += (alpha[n] * gset[n] + Math.max(0, C * gset[n]));
				G += Math.abs(gset[n]) * (C - Math.abs(alpha[n]));
			}
		}
		return G;
	}

	private double computeGapTarget() {

		double G = 0;
		double mu = 0;
		int l = 0;
		for (int n = 0; n < train.size(); n++) {
			if (keset[n]) {
				mu += gset[n];
				l++;
			}
		}
		if (l == 0)
			mu = 0;
		else
			mu = mu * mu / l;

		for (int n = 0; n < train.size(); n++) {
			if (keset[n]) {
				G += gset[n] * gset[n];
			}
		}

		return Math.sqrt(G - mu);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.classifier.Classifier#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(T e) {
		double r = 0;
		for (int n = 0; n < train.size(); n++) {
			if (keset[n])
				r += alpha[n] * kernel.valueOf(train.get(n).sample, e);
		}
		return r;
	}

	/**
	 * Tells the C hyperparameter
	 * 
	 * @return C
	 */
	public double getC() {
		return C;
	}

	/**
	 * Sets the C hyperparameter (default 1.0)
	 * 
	 * @param c C
	 */
	public void setC(double c) {
		C = c;
	}

	/**
	 * Tells the number of epochs of training (default 2)
	 * 
	 * @return E
	 */
	public long getE() {
		return E;
	}

	/**
	 * Sets the number of epoch of training (default 2)
	 * 
	 * @param e E
	 */
	public void setE(long e) {
		E = e;
	}

	/**
	 * Tells support vectors coefficients in order of the training list
	 * 
	 * @return alpha
	 */
	public double[] getAlphas() {
		double[] a = new double[alpha.length];
		for (int s = 0; s < a.length; s++)
			a[s] = alpha[s] * train.get(s).label;
		return a;
	}

	/**
	 * Set the kernel to use
	 * 
	 * @param kernel the kernel
	 */
	public void setKernel(Kernel<T> kernel) {
		this.kernel = kernel;
	}

	/**
	 * Tells the parameter s of the ramp loss (default -1)
	 * 
	 * @return s
	 */
	public double getS() {
		return s;
	}

	/**
	 * Sets the parameter s of the ramp loss (default -1)
	 * 
	 * @param s s
	 */
	public void setS(double s) {
		this.s = s;
	}

	/**
	 * Creates and returns a copy of this object.
	 * 
	 * @see java.lang.Object#clone()
	 */
	@SuppressWarnings("unchecked")
	@Override
	public LaSVMI<T> copy() throws CloneNotSupportedException {
		return (LaSVMI<T>) super.clone();
	}

	/**
	 * Returns the kernel used by this classifier
	 * @return the kernel
	 */
	public Kernel<T> getKernel() {
		return kernel;
	}

}
