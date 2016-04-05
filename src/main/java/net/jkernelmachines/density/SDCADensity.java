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

import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import net.jkernelmachines.kernel.Kernel;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.algebra.VectorOperations;

/**
 * One Class SVM estimation using Stochastic Dual Coordinate Ascent.
 * 
 * @author picard
 * 
 */
public class SDCADensity<T> implements DensityFunction<T> {

	private static final long serialVersionUID = -9091693938716912183L;

	Kernel<T> kernel;
	T[] samples;
	double[] alphas;
	double[][] km;
	public boolean CACHED_KERNEL = true;

	double C = 1.0;
	int E = 10;

	List<T> train;

	/**
	 * @param k
	 */
	public SDCADensity(Kernel<T> k) {
		kernel = k;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.density.DensityFunction#train(java.lang.Object)
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void train(T e) {
		if (train == null) {
			train = new ArrayList<T>();
			train.add(e);
			train(train);
		}
		else {
			train.add(e);
			int n = train.size();
			samples = (T[]) new Object[n];
			if(CACHED_KERNEL) {
				ArrayList<TrainingSample<T>> t_list = new ArrayList<TrainingSample<T>>();
				for (T t : train) {
					t_list.add(new TrainingSample<T>(t, 1));
				}
				km = kernel.getKernelMatrix(t_list);
			}
			else {
				km = null;
			}
			train.toArray(samples);
			double[] a2 = Arrays.copyOf(alphas, n);
			a2[n-1] = 0;
			alphas = a2;
			update(n-1);
		}

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.density.DensityFunction#train(java.util.List)
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void train(List<T> l) {
		int n = l.size();
		train = new ArrayList<T>(n);
		train.addAll(l);

		if (CACHED_KERNEL) {
			ArrayList<TrainingSample<T>> t_list = new ArrayList<TrainingSample<T>>();
			for (T t : l) {
				t_list.add(new TrainingSample<T>(t, 1));
			}
			km = kernel.getKernelMatrix(t_list);
		} else
			km = null;

		samples = (T[]) new Object[n];
		l.toArray(samples);

		alphas = new double[n];

		List<Integer> indices = new ArrayList<Integer>(n);
		for (int i = 0; i < n; i++) {
			indices.add(i);
		}

		for (int e = 0; e < E; e++) {
			Collections.shuffle(indices);
			for (int i = 0; i < n; i++) {
				update(indices.get(i));
			}
		}

	}

	/**
	 * dual variable update
	 * 
	 * @param i
	 *            index f the dual variable
	 */
	private final void update(int i) {
		double[] kmi;
		if (km!= null && CACHED_KERNEL) {
			kmi = km[i];
		} else {
			kmi = new double[samples.length];
			for (int j = 0; j < samples.length; j++)
				kmi[j] = kernel.valueOf(samples[i], samples[j]);
		}
		double z = VectorOperations.dot(alphas, kmi);
		double kmii = 0;
		if (km == null)
			kmii = kernel.valueOf(samples[i], samples[i]);
		else
			kmii = km[i][i];
//		double suma = 0;
//		for(int d = 0 ; d < alphas.length ; d++)
//			suma += alphas[d];
		double da = (1. - z) / kmii + alphas[i];// - C/samples.length*(suma - 1);
		alphas[i] = max(0, min(C, da));

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.density.DensityFunction#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(T e) {
		double sum = 0;
		for (int i = 0; i < samples.length; i++)
			sum += alphas[i] * kernel.valueOf(samples[i], e);
		return max(0, min(1, sum)) ;
	}

	public Kernel<T> getKernel() {
		return kernel;
	}

	public void setKernel(Kernel<T> kernel) {
		this.kernel = kernel;
	}

	public boolean isCACHED_KERNEL() {
		return CACHED_KERNEL;
	}

	public void setCACHED_KERNEL(boolean cACHED_KERNEL) {
		CACHED_KERNEL = cACHED_KERNEL;
	}

	public double getC() {
		return C;
	}

	public void setC(double c) {
		C = c;
	}

	public int getE() {
		return E;
	}

	public void setE(int e) {
		E = e;
	}

	public double[] getAlphas() {
		return alphas;
	}

}
