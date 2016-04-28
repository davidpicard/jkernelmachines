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

import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import net.jkernelmachines.kernel.Kernel;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.type.TrainingSampleStream;
import net.jkernelmachines.util.algebra.VectorOperations;

/**
 * <p>
 * SDCA svm algorithm from Shalev-Shwartz.
 * </p>
 * <p>
 * Stochastic Dual Coordinate Ascent Methods for Regularized Loss Minimization,
 * 
 * Shai Shalev-Shwartz, Tong Zhang
 * JMLR, 2013.
 * </p>
 * 
 * @author picard
 * 
 */
public class SDCA<T> implements KernelSVM<T>, OnlineClassifier<T> {

	Kernel<T> kernel;
	T[] samples;
	int[] labels;
	double[] alphas;

	double C = 1.0;
	int E = 5;

	List<TrainingSample<T>> train;

	// tmp variables
	private int n;
	private double[][] km;
	private boolean cacheKernel = true;

	/**
	 * @param kernel kernel
	 */
	public SDCA(Kernel<T> kernel) {
		super();
		this.kernel = kernel;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.classifier.Classifier#train(fr.lip6.jkernelmachines
	 * .type.TrainingSample)
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void train(TrainingSample<T> t) {
		if (train == null) {
			train = new ArrayList<TrainingSample<T>>();
			alphas = new double[0];
			samples = (T[]) new Object[0];
		}
		if(train.contains(t)) {
			updateNoCache(train.indexOf(t));
		}
		else {
			train.add(t);
			alphas = Arrays.copyOf(alphas, alphas.length+1);
			samples = Arrays.copyOf(samples, samples.length+1);
			samples[samples.length-1] = t.sample;
			updateNoCache(train.size()-1);
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#train(java.util.List)
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void train(List<TrainingSample<T>> l) {
		n = l.size();
		train = new ArrayList<TrainingSample<T>>(n);
		train.addAll(l);

		if (cacheKernel && l.size()*l.size()*8 < Runtime.getRuntime().freeMemory()) {
			km = kernel.getKernelMatrix(train);
		}

		samples = (T[]) new Object[n];
		labels = new int[n];

		for (int i = 0; i < n; i++) {
			TrainingSample<T> t = l.get(i);
			samples[i] = t.sample;
			labels[i] = t.label;
		}

		alphas = new double[n];

		List<Integer> indices = new ArrayList<Integer>(n);
		for (int i = 0; i < n; i++) {
			indices.add(i);
		}

		if (cacheKernel) {
			for (int e = 0; e < E; e++) {
				Collections.shuffle(indices);
				for (int i = 0; i < n; i++) {
					update(indices.get(i));
				}
			}
		} else {
			for (int e = 0; e < E; e++) {
				Collections.shuffle(indices);
				for (int i = 0; i < n; i++) {
					updateNoCache(indices.get(i));
				}
			}
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.classifier.OnlineClassifier#onlineTrain(fr.lip6
	 * .jkernelmachines.type.TrainingSampleStream)
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void onlineTrain(TrainingSampleStream<T> stream) {
		if(train == null || alphas == null) {
			train = new ArrayList<>();
			alphas = new double[0];
			samples = (T[])new Object[0];
		}
		// remove caching Gram matrix
		cacheKernel = false;
		TrainingSample<T> t;
		int i = 0;
		while((t = stream.nextSample()) != null) {
			i++;
			if(train.contains(t)) {
				updateNoCache(train.indexOf(t));
			}
			else {
				train.add(t);
				alphas = Arrays.copyOf(alphas, alphas.length+1);
				samples = Arrays.copyOf(samples, samples.length+1);
				samples[samples.length-1] = t.sample;
				updateNoCache(train.size()-1);
			}
			System.out.print("\r"+i);
		}
		System.out.println("\r");
	}

	/**
	 * dual variable update
	 * 
	 * @param i
	 *            index f the dual variable
	 */
	private final void update(int i) {
		double y = labels[i];
		double z = (VectorOperations.dot(alphas, km[i]));
		double da = (1 - y * z) / km[i][i] + y * alphas[i];
		alphas[i] = y * max(0, min(C, da));

	}

	/**
	 * dual variable update
	 * 
	 * @param i
	 *            index f the dual variable
	 */
	private final void updateNoCache(int i) {
		TrainingSample<T> sam = train.get(i);
		double y = sam.label;
		double z = valueOf(sam.sample);
		double da = (1 - y * z) / kernel.valueOf(sam.sample, sam.sample) + y * alphas[i];
		alphas[i] = y * max(0, min(C, da));

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.classifier.Classifier#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(T e) {
		double z = 0;
		for (int i = 0; i < alphas.length; i++) {
			z += alphas[i] * kernel.valueOf(samples[i], e);
		}
		return z;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#copy()
	 */
	@SuppressWarnings("unchecked")
	@Override
	public Classifier<T> copy() throws CloneNotSupportedException {
		return (SDCA<T>) this.clone();
	}

	/**
	 * Get the number of epochs to train the classifier
	 * 
	 * @return the number of epochs
	 */
	public int getE() {
		return E;
	}

	/**
	 * Set the number of epochs (going through all samples once) for the
	 * training phase
	 * 
	 * @param e
	 *            the number of epochs
	 */
	public void setE(int e) {
		E = e;
	}

	@Override
	public void setKernel(Kernel<T> k) {
		this.kernel = k;
	}

	@Override
	public double[] getAlphas() {
		double[] a = new double[alphas.length];
		for (int s = 0; s < a.length; s++)
			a[s] = alphas[s] * train.get(s).label;
		return a;
	}

	@Override
	public void setC(double c) {
		C = c;
	}

	@Override
	public double getC() {
		return C;
	}

	public double getObjective() {
		double obj = 0;

		// norm of w
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				obj += 0.5 * alphas[i] * alphas[j] * (km[i][j]);
			}
		}

		// loss
		for (int i = 0; i < n; i++) {
			double v = valueOf(samples[i]);
			obj += C * max(0, 1 - labels[i] * v);
		}

		return obj / (double) n;
	}

	public double getDualObjective() {
		double obj = 0;

		for (int i = 0; i < n; i++) {
			obj += labels[i] * alphas[i];
		}

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				obj -= 0.5 * alphas[i] * alphas[j] * km[i][j];

			}
		}

		return obj / (double) n;
	}

	@Override
	public Kernel<T> getKernel() {
		return kernel;
	}

	public boolean isCacheKernel() {
		return cacheKernel;
	}

	public void setCacheKernel(boolean cacheKernel) {
		this.cacheKernel = cacheKernel;
	}

}
