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
package net.jkernelmachines.projection;

import static net.jkernelmachines.util.algebra.MatrixOperations.eig;
import static net.jkernelmachines.util.algebra.MatrixOperations.transi;

import java.util.ArrayList;
import java.util.List;

import net.jkernelmachines.kernel.Kernel;
import net.jkernelmachines.type.TrainingSample;

/**
 * Kernel principal component analysis, using generic datatypes.
 * 
 * @author picard
 * 
 */
public class KernelPCA<T> {

	private Kernel<T> kernel;
	private double mean = 0;
	private double[][] projectors;
	private double[] whiteningCoefficients;
	private List<TrainingSample<T>> list;
	private int dim;

	public KernelPCA(Kernel<T> k) {
		this.kernel = k;
	}

	public void train(List<TrainingSample<T>> list) {
		this.list = list;

		// SVD of kernel matrix
		double[][] K = kernel.getKernelMatrix(list);
		mean = 0;
		for (int i = 0; i < K.length; i++) {
			for (int j = i; j < K.length; j++) {
				if (i == j) {
					mean += K[i][j];
				} else {
					mean += 2 * K[i][j];
				}
			}
		}
		mean /= (K.length * K.length);
		for (int i = 0; i < K.length; i++) {
			for (int j = i; j < K.length; j++) {
				K[i][j] -= mean;
				K[j][i] = K[i][j];
			}
		}
		double[][][] eig = eig(K);
		dim = eig[0].length;

		// projectors
		projectors = transi(eig[0]);

		// whitening coeff
		whiteningCoefficients = new double[dim];
		for (int d = 0; d < dim; d++) {
			double p = eig[1][d][d];
			if (p > 1e-15) {
				whiteningCoefficients[d] = 1. / Math.sqrt(p);
			}
		}
	}

	public TrainingSample<double[]> project(TrainingSample<T> t,
			boolean whitening) {
		double[] proj = new double[dim];

		for (int i = 0; i < list.size(); i++) {
			double v = (kernel.valueOf(list.get(i).sample, t.sample) - mean);
			for (int d = 0; d < dim; d++) {
				proj[d] += projectors[d][i] * v;
			}
		}
		
		for (int d = 0; d < dim; d++) {
			proj[d] *= whiteningCoefficients[d];
			if (whitening) {
				proj[d] *= whiteningCoefficients[d];
			}
		}

		return new TrainingSample<double[]>(proj, t.label);
	}

	public TrainingSample<double[]> project(TrainingSample<T> t) {
		return project(t, false);
	}

	/**
	 * Performs the projection on a list of samples.
	 * 
	 * @param list
	 *            the list of input samples
	 * @return a new list with projected samples
	 */
	public List<TrainingSample<double[]>> projectList(
			final List<TrainingSample<T>> list) {
		List<TrainingSample<double[]>> out = new ArrayList<TrainingSample<double[]>>();

		for (TrainingSample<T> t : list) {
			out.add(project(t));
		}

		return out;
	}

	/**
	 * Performs the projection on a list of samples with optional whitening
	 * (unitary covariance matrix).
	 * 
	 * @param list
	 *            the list of input samples
	 * @param whitening
	 *            option to perform a whitened projection
	 * @return a new list with projected samples
	 */
	public List<TrainingSample<double[]>> projectList(
			final List<TrainingSample<T>> list, boolean whitening) {
		List<TrainingSample<double[]>> out = new ArrayList<TrainingSample<double[]>>();

		for (TrainingSample<T> t : list) {
			out.add(project(t, whitening));
		}

		return out;
	}

	/**
	 * Get the kernel use in the KernelPCA
	 * 
	 * @return the current kernel
	 */
	public Kernel<T> getKernel() {
		return kernel;
	}

	/**
	 * Set the current Kernel
	 * 
	 * @param kernel
	 *            the kernel
	 */
	public void setKernel(Kernel<T> kernel) {
		this.kernel = kernel;
	}

	/**
	 * Get the mean of the current Kernel
	 * 
	 * @return the mean of the kernel over the training set
	 */
	public double getMean() {
		return mean;
	}

	/**
	 * Get the projector coefficients obtained after learning
	 * 
	 * @return the kernel expansion coefficients
	 */
	public double[][] getProjectors() {
		return projectors;
	}

	/**
	 * Get the whitening coefficient
	 * 
	 * @return the whitening coefficients
	 */
	public double[] getWhiteningCoefficients() {
		return whiteningCoefficients;
	}

}
