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
package fr.lip6.jkernelmachines.density;

import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import fr.lip6.jkernelmachines.kernel.Kernel;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

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
	int E = 100;

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
	@Override
	public void train(T e) {
		if (train == null)
			train = new ArrayList<T>();
		train.add(e);
		train(train);

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
		C = 1. / n;

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
		if (CACHED_KERNEL) {
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
		double da = (1 - z) / kmii + alphas[i];
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
		return sum;
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
