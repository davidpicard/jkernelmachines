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
package fr.lip6.jkernelmachines.classifier;

import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import fr.lip6.jkernelmachines.kernel.Kernel;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

/**
 * <p>
 * SDCA svm algorithm from Shalev-Shwartz.
 * </p>
 * <p>
 * Stochastic Dual Coordinate Ascent Methods for Regularized Loss Minimization,
 * <br/>
 * Shai Shalev-Shwartz, Tong Zhang<br/>
 * JMLR, 2013.
 * </p>
 * 
 * @author picard
 * 
 */
public class SDCA<T> implements KernelSVM<T> {

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

	/**
	 * @param kernel
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
	@Override
	public void train(TrainingSample<T> t) {
		if(train == null) {
			train = new ArrayList<TrainingSample<T>>();
		}
		train.add(t);
		train(train);
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

		km = kernel.getKernelMatrix(train);

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
		double y = labels[i];
		double z = (VectorOperations.dot(alphas, km[i]));
		double da = (1 - y * z) / km[i][i] + y * alphas[i];
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
}
