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
package net.jkernelmachines.kernel;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import net.jkernelmachines.threading.ThreadedMatrixOperator;
import net.jkernelmachines.type.TrainingSample;

/**
 * Base class for kernels
 * 
 * @author dpicard
 * 
 * @param <T>
 *            Data type of input space
 */
public abstract class Kernel<T> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1663774351688566794L;

	public String name = "k_default";

	/**
	 * compute the kernel similarity between two element of input space
	 * 
	 * @param t1
	 *            first element
	 * @param t2
	 *            second element
	 * @return the kernel value
	 */
	public abstract double valueOf(T t1, T t2);

	/**
	 * kernel similarity to self
	 * 
	 * @param t1
	 *            the element to compute the similarity to itself
	 * @return the norm of t1
	 */
	public abstract double valueOf(T t1);

	/**
	 * kernel similarity normalized such that k(t1, t1) = 1
	 * 
	 * @param t1
	 *            first element
	 * @param t2
	 *            second element
	 * @return normalized similarity
	 */
	public double normalizedValueOf(T t1, T t2) {
		return valueOf(t1, t2) / Math.sqrt(valueOf(t1, t1) * valueOf(t2, t2));
	}

	/**
	 * return the Gram Matrix of this kernel computed on given samples
	 * 
	 * @param l
	 *            list of samples on which to compute the Gram matrix
	 * @return double[][] containing similarities in the order of the list.
	 */
	public double[][] getKernelMatrix(final List<TrainingSample<T>> l) {
		double[][] matrix = new double[l.size()][l.size()];

		// computing matrix
		ThreadedMatrixOperator factory = new ThreadedMatrixOperator() {
			@Override
			public void doLines(double[][] matrix, int from, int to) {
				for (int index = from; index < to; index++) {
					T s1 = l.get(index).sample;
					for (int j = index; j < matrix.length; j++) {
						matrix[index][j] = valueOf(s1, l.get(j).sample);
						matrix[j][index] = matrix[index][j];
					}
				}
			}
		};

		factory.getMatrix(matrix);

		return matrix;
	}

	/**
	 * return the Gram Matrix of this kernel computed on given samples, with
	 * similarities of one element to itself normalized to one.
	 * 
	 * @param e
	 *            the list of samples
	 * @return double[][] containing similarities in the order of the the list.
	 */
	public double[][] getNormalizedKernelMatrix(ArrayList<TrainingSample<T>> e) {
		double[][] matrix = new double[e.size()][e.size()];
		for (int i = 0; i < e.size(); i++) {
			for (int j = i; j < e.size(); j++) {
				matrix[i][j] = normalizedValueOf(e.get(i).sample,
						e.get(j).sample);
				matrix[j][i] = matrix[i][j];
			}
		}

		return matrix;
	}

	/**
	 * returns a vector containing the similarity values of a given sample to a
	 * list of samples (i.e. a line of a kernel matrix if said sample is in the
	 * list)
	 * 
	 * @param x
	 *            sample
	 * @param l
	 *            list of sample
	 * @return [k(x, l[i])]_i
	 */
	public double[] getKernelMatrixLine(T x, List<TrainingSample<T>> l) {
		double[] ki = new double[l.size()];
		for (int i = 0; i < l.size(); i++) {
			ki[i] = valueOf(x, l.get(i).sample);
		}
		return ki;
	}

	/**
	 * Set the name of this kernel
	 * 
	 * @param n the name of the kernel
	 */
	public void setName(String n) {
		name = n;
	}

	/**
	 * return the name of this kernel
	 */
	public String toString() {
		return name;
	}

}
