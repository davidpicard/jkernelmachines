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
package net.jkernelmachines.classifier.multiclass;

import static java.lang.Math.abs;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import net.jkernelmachines.classifier.Classifier;
import net.jkernelmachines.classifier.KernelSVM;
import net.jkernelmachines.kernel.Kernel;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.algebra.MatrixOperations;
import net.jkernelmachines.util.algebra.MatrixVectorOperations;
import net.jkernelmachines.util.algebra.VectorOperations;

/**
 * 
 * <p>
 * This is a straight forward extension of SDCA svm algorithm from
 * Shalev-Shwartz to multiclass using a multiclass loss function.
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
public class MulticlassSDCA<T> implements MulticlassClassifier<T>, KernelSVM<T> {

	Kernel<T> kernel;
	List<TrainingSample<T>> tlist;
	double[][] alpha;

	int nb_classes;
	List<Integer> classes;

	double C = 1.;
	double E = 25;

	public MulticlassSDCA(Kernel<T> k) {
		this.kernel = k;
		nb_classes = 0;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#train(java.util.List)
	 */
	@Override
	public void train(List<TrainingSample<T>> l) {

		tlist = new ArrayList<TrainingSample<T>>();
		tlist.addAll(l);

		// kernel matrix
		double[][] k_matrix = kernel.getKernelMatrix(tlist);

		nb_classes = 0;
		classes = new ArrayList<Integer>();
		List<Integer> perm = new ArrayList<Integer>();
		int ts = 0;
		for (TrainingSample<T> t : tlist) {
			if (!classes.contains(t.label)) {
				nb_classes++;
				classes.add(t.label);
			}
			perm.add(ts++);
		}

		alpha = new double[tlist.size()][nb_classes];
		double[][] alphaTrans = new double[nb_classes][tlist.size()];
		double[] values = new double[nb_classes];
		for (int e = 0; e < E * nb_classes; e++) {
			Collections.shuffle(perm);
			for (int i : perm) {
				TrainingSample<T> t = tlist.get(i);
				int ty = classes.indexOf(t.label);

				// early bail out if correct class
				MatrixVectorOperations
						.rMuli(values,
								MatrixOperations.transi(alphaTrans, alpha),
								k_matrix[i]);
				boolean stop = true;
				for (int y = 0; y < nb_classes; y++) {
					if (y != ty && values[y] >= values[ty]) {
						stop = false;
					}
				}
				if (stop) {
					continue;
				}

				// init grad
				double[] g = new double[nb_classes];
				Arrays.fill(g, 1);

				// full grad
				double[] kij_aj = new double[nb_classes];
				for (int j = 0; j < tlist.size(); j++) {
					VectorOperations.muli(kij_aj, alpha[j], k_matrix[i][j]);
					VectorOperations.addi(g, g, 1.0, kij_aj);
				}
				VectorOperations.muli(g, g, -1.0 / k_matrix[i][i]);
				g[ty] = 0;

				// box constraints on gradient
				double sum = 0;
				for (int y = 0; y < nb_classes; y++) {
					if (y != ty) {
						if (alpha[i][y] >= 0 && g[y] > 0) {
							g[y] = 0;
						}
						if (alpha[i][y] + g[y] > 0) {
							g[y] = -alpha[i][y];
						}
						sum += g[y];
					}
				}
				// box on ty
				if (alpha[i][ty] - sum > C) {
					// System.out.println("box 2 sum: "+sum+" aiy: "+alpha[i][ty]);
					for (int y = 0; y < nb_classes; y++) {
						if (y != ty) {
							g[y] *= (alpha[i][ty] - C) / sum;
						}
					}
					sum = alpha[i][ty] - C;
				}
				g[ty] = -sum;

				// System.out.println("g: "+Arrays.toString(g)+" ty: "+ty);

				// update
				double[] a_new = VectorOperations.add(alpha[i], 1, g);

				// num cleaning
				for (int y = 0; y < nb_classes; y++) {
					if (y != ty) {
						if (a_new[y] > 0) {
							a_new[y] = 0;
						}
					}
				}

				sum = 0;
				for (int d = 0; d < nb_classes; d++) {
					sum += a_new[d];
				}
				if (abs(sum) > 1e-10) {
					System.out.println("error with " + i + " sum(a)= " + sum
							+ " a: " + Arrays.toString(a_new));
					System.exit(0);
				}

				// save
				alpha[i] = a_new;
			}
		}

		// System.out.println("alpha: "+Arrays.deepToString(alpha));

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.classifier.Classifier#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(T e) {
		if (nb_classes <= 0) {
			return 0;
		}
		double[] v = new double[nb_classes];
		for (int j = 0; j < tlist.size(); j++) {
			VectorOperations.addi(v, v, kernel.valueOf(e, tlist.get(j).sample),
					alpha[j]);
		}
		int idmax = 0;
		double vmax = Double.NEGATIVE_INFINITY;
		for (int d = 0; d < nb_classes; d++) {
			if (v[d] > vmax) {
				idmax = d;
				vmax = v[d];
			}
		}
		return classes.get(idmax);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#copy()
	 */
	@SuppressWarnings("unchecked")
	@Override
	public Classifier<T> copy() throws CloneNotSupportedException {
		return (MulticlassSDCA<T>) this.clone();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.classifier.multiclass.MulticlassClassifier#
	 * getConfidence(java.lang.Object)
	 */
	@Override
	public double getConfidence(T t) {
		if (nb_classes <= 0) {
			return 0;
		}
		double[] v = new double[nb_classes];
		for (int j = 0; j < tlist.size(); j++) {
			VectorOperations.addi(v, v, kernel.valueOf(t, tlist.get(j).sample),
					alpha[j]);
		}
		double vmax = Double.NEGATIVE_INFINITY;
		for (int d = 0; d < nb_classes; d++) {
			if (v[d] > vmax) {
				vmax = v[d];
			}
		}
		return vmax;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.classifier.multiclass.MulticlassClassifier#
	 * getConfidences(java.lang.Object)
	 */
	@Override
	public Map<Integer, Double> getConfidences(T t) {
		if (nb_classes <= 0) {
			return null;
		}
		double[] v = new double[nb_classes];
		for (int j = 0; j < tlist.size(); j++) {
			VectorOperations.addi(v, v, kernel.valueOf(t, tlist.get(j).sample),
					alpha[j]);
		}

		HashMap<Integer, Double> map = new HashMap<>();
		for (int i = 0; i < v.length; i++) {
			map.put(classes.get(i), v[i]);
		}
		return map;
	}

	@Override
	public Kernel<T> getKernel() {
		return kernel;
	}

	@Override
	public void setKernel(Kernel<T> kernel) {
		this.kernel = kernel;
	}

	@Override
	public double getC() {
		return C;
	}

	@Override
	public void setC(double c) {
		C = c;
	}

	/**
	 * return the number of epochs
	 * 
	 * @return
	 */
	public double getE() {
		return E;
	}

	/**
	 * Sets the number of epochs
	 * 
	 * @param e
	 */
	public void setE(double e) {
		E = e;
	}

	@Override
	public double[] getAlphas() {
		throw new RuntimeException("operation not possible");
	}

	/**
	 * Returns the matrix of dual variables in the order [sample, class]
	 * 
	 * @return
	 */
	public double[][] getMulticlassAlphas() {
		return alpha;
	}

}
