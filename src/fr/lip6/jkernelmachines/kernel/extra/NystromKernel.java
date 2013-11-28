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
package fr.lip6.jkernelmachines.kernel.extra;

import static java.lang.Math.min;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import fr.lip6.jkernelmachines.kernel.Kernel;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.DebugPrinter;
import fr.lip6.jkernelmachines.util.algebra.MatrixOperations;
import fr.lip6.jkernelmachines.util.algebra.ThreadedMatrixOperations;

/**
 * This kernel provides a fast approximation of a given kernel using the Nystrom
 * approximation.
 * <p>
 * A fast active learning algorithm analog to that used in:<br/>
 * <strong>Fast Approximation of Distance Between Elastic Curves using
 * Kernels</strong><br />
 * Hedi Tabia; David Picard; Hamid Laga; Philippe-Henri Gosselin<br/>
 * British Machine Vision Conference, Sep 2013, United Kingdom. British Machine
 * Vision Conference<br />
 * </p>
 * 
 * @author picard
 * 
 */
public class NystromKernel<T> extends Kernel<T> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6324067887253049771L;

	private static DebugPrinter debug = new DebugPrinter();

	Kernel<T> kernel;
	DoubleLinear linear;
	private double[][] projectors;
	private double[] eigenvalues;
	private List<TrainingSample<T>> list;
	private int dim = -1;

	/**
	 * Default constructor with kernel to approximate as argument
	 * 
	 * @param kernel
	 *            the kernel to be approximated
	 */
	public NystromKernel(Kernel<T> kernel) {
		this.kernel = kernel;
		linear = new DoubleLinear();
	}

	/**
	 * Train the Nystrom approx on a full training set.
	 * 
	 * <p>
	 * Might be costly for large training sets, since it involves the inversion
	 * of the Gram matrix.
	 * </p>
	 * 
	 * @param list
	 *            the training list of samples
	 */
	public void train(List<TrainingSample<T>> list) {
		this.list = new ArrayList<TrainingSample<T>>();
		this.list.addAll(list);
		dim = list.size();
		debug.println(3, "matrix size : " + dim);
		double[][] matrix = kernel.getKernelMatrix(list);
		double[][][] eig = MatrixOperations.eig(matrix);

		projectors = ThreadedMatrixOperations.transi(eig[0]);
		eigenvalues = new double[dim];
		for (int d = 0; d < dim; d++) {
			double p = eig[1][d][d];
			if (p > 1e-15) {
				eigenvalues[d] = 1. / Math.sqrt(p);
			}
		}

	}

	public void activeTrain(List<TrainingSample<T>> list, int iterations,
			int samples, int pool) {
		this.list = new ArrayList<TrainingSample<T>>();
		List<TrainingSample<T>> dup = new ArrayList<TrainingSample<T>>();
		List<TrainingSample<T>> poolList = new ArrayList<TrainingSample<T>>();
		dup.addAll(list);

		for (int i = 0; i < iterations; i++) {
			debug.println(3, "active iteration " + i);
			// select pool samples randomly
			Collections.shuffle(dup);

			poolList.clear();
			int dupSize = dup.size();
			for (int j = 0; j < min(pool, dupSize); j++) {
				poolList.add(dup.remove(0));
			}

			// computes errors
			class Err {
				public TrainingSample<T> t;
				public double error;
			}
			List<Err> errors = new ArrayList<Err>();
			List<TrainingSample<T>> subPool = poolList.subList(0,
					min(pool, poolList.size()));
			double[][] mori = kernel.getKernelMatrix(subPool);
			double[][] mapp = getKernelMatrix(subPool);

			for (int j = 0; j < min(pool, poolList.size()); j++) {
				TrainingSample<T> t = poolList.get(j);
				double e = 0;
				for (int k = 0; k < min(pool, poolList.size()); k++) {
					double v1 = mori[j][k];
					double v2 = mapp[j][k];
					double d = (v1 - v2);
					e += d * d;
				}
				Err err = new Err();
				err.t = t;
				err.error = e;
				errors.add(err);
			}
			Collections.sort(errors, new Comparator<Err>() {
				@Override
				public int compare(Err t1, Err t2) {
					return -Double.compare(t1.error, t2.error);
				}
			});

			// add high error to the list
			for (int k = 0; k < samples; k++) {
				this.list.add(errors.remove(0).t);
			}
			// add low error samples back to the pool
			while (!errors.isEmpty()) {
				dup.add(errors.remove(0).t);
			}

			// train
			this.train(this.list);
		}

	}

	/**
	 * Project a sample to the space induced by the Nystrom approx
	 * 
	 * @param sample
	 * @return
	 */
	public double[] projectSample(T sample) {
		// if was not train, return 0
		if (dim <= 0)
			return new double[1];

		double[] out = new double[dim];
		double[] km = new double[list.size()];
		for (int i = 0; i < list.size(); i++)
			km[i] = kernel.valueOf(list.get(i).sample, sample);

		for (int d = 0; d < dim; d++) {
			for (int i = 0; i < list.size(); i++) {
				out[d] += projectors[d][i] * km[i];
			}
			out[d] *= eigenvalues[d];
		}

		return out;
	}

	public List<TrainingSample<double[]>> projectList(List<TrainingSample<T>> l) {
		List<TrainingSample<double[]>> out = new ArrayList<TrainingSample<double[]>>(
				l.size());
		for (TrainingSample<T> t : l) {
			double[] d = projectSample(t.sample);
			out.add(new TrainingSample<double[]>(d, t.label));
		}
		return out;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.kernel.Kernel#valueOf(java.lang.Object,
	 * java.lang.Object)
	 */
	@Override
	public double valueOf(T t1, T t2) {
		return linear.valueOf(projectSample(t1), projectSample(t2));
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.kernel.Kernel#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(T t1) {
		double[] s = projectSample(t1);
		return linear.valueOf(s, s);
	}

}
