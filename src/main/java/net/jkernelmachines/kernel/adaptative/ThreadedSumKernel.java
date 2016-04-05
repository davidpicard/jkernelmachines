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
package net.jkernelmachines.kernel.adaptative;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

import net.jkernelmachines.kernel.Kernel;
import net.jkernelmachines.threading.ThreadPoolServer;
import net.jkernelmachines.threading.ThreadedMatrixOperator;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.DebugPrinter;
import net.jkernelmachines.util.algebra.VectorOperations;

/**
 * Major kernel computed as a weighted sum of minor kernels : K = w_i * k_i
 * Computation of the kernel matrix is done by running a thread on sub matrices.
 * The number of threads is chosen as function of the number of available cpus.
 * 
 * @author dpicard
 *
 * @param <T>
 */
public class ThreadedSumKernel<T> extends Kernel<T> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7780445301175174296L;

	private Map<Kernel<T>, Double> kernels;
	private DebugPrinter debug = new DebugPrinter();
	

	public ThreadedSumKernel() {
		kernels = new HashMap<Kernel<T>, Double>();
	}

	/**
	 * Sets the weights to h. Beware! It does not make a copy of h!
	 * 
	 * @param h
	 */
	public ThreadedSumKernel(Map<Kernel<T>, Double> h) {
		// kernels = h;
		kernels = new HashMap<Kernel<T>, Double>();
		kernels.putAll(h);
	}

	/**
	 * adds a kernel to the sum with weight 1.0
	 * 
	 * @param k
	 */
	public void addKernel(Kernel<T> k) {
		synchronized (kernels) {
			kernels.put(k, 1.0);
		}
	}

	/**
	 * adds a kernel to the sum with weight d
	 * 
	 * @param k
	 * @param d
	 */
	public void addKernel(Kernel<T> k, double d) {
		synchronized (kernels) {
			kernels.put(k, d);
		}
	}

	/**
	 * removes kernel k from the sum
	 * 
	 * @param k
	 */
	public void removeKernel(Kernel<T> k) {
		synchronized (kernels) {
			kernels.remove(k);
		}
	}

	/**
	 * gets the weights of kernel k
	 * 
	 * @param k
	 * @return the weight associated with k
	 */
	public double getWeight(Kernel<T> k) {
		synchronized (kernels) {
			if (kernels.containsKey(k))
				return kernels.get(k);
			return 0;
		}
	}

	/**
	 * Sets the weight of kernel k
	 * 
	 * @param k
	 * @param d
	 */
	public void setWeight(Kernel<T> k, Double d) {
		kernels.put(k, d);
	}

	@Override
	public double valueOf(T t1, T t2) {
		double sum = 0.;
		for (Kernel<T> k : kernels.keySet()) {
			double w = kernels.get(k);
			if (w != 0)
				sum += k.valueOf(t1, t2) * kernels.get(k);
		}

		return sum;
	}

	@Override
	public double valueOf(T t1) {
		return valueOf(t1, t1);
	}

	/**
	 * get the list of kernels and associated weights.
	 * 
	 * @return hashtable containing kernels as keys and weights as values.
	 */
	public Map<Kernel<T>, Double> getWeights() {
		return kernels;
	}

	@Override
	public double[][] getKernelMatrix(List<TrainingSample<T>> list) {
		final List<TrainingSample<T>> l = list;
		// init matrix with zeros
		double matrix[][] = new double[l.size()][l.size()];

		for (final Kernel<T> k : kernels.keySet()) {
			final double w = kernels.get(k);

			// check w
			if (w == 0)
				continue;

			final double[][] m = k.getKernelMatrix(l);
			// specific factory
			ThreadedMatrixOperator tmo = new ThreadedMatrixOperator() {

				@Override
				public void doLines(double[][] matrix, int from, int to) {
					for (int index = from; index < to; index++) {
						for (int j = index; j < matrix[index].length; j++) {
							matrix[index][j] += m[index][j] * w;
							matrix[j][index] = matrix[index][j];
						}
					}
				};

			};

			matrix = tmo.getMatrix(matrix);
		}

		return matrix;
	}

	@Override
	public double[] getKernelMatrixLine(T x, List<TrainingSample<T>> list) {
		final List<TrainingSample<T>> l = list;
		final T t = x;

		final double output[] = new double[l.size()];

		ThreadPoolExecutor te = ThreadPoolServer.getThreadPoolExecutor();
		Queue<Future<?>> futures = new LinkedList<>();

		for (final Kernel<T> k : kernels.keySet()) {
			final double w = kernels.get(k);

			futures.add(te.submit(new Runnable() {

				@Override
				public void run() {
					double[] line = k.getKernelMatrixLine(t, l);
					synchronized (output) {
						VectorOperations.addi(output, output, w, line);
					}
				}
			}));
		}

		try {
			// wait for all jobs
			while (!futures.isEmpty()) {
				futures.remove().get();
			}
		} catch (InterruptedException e) {

			debug.println(3, "ThreadedSumKernel : getKernelMatrixLine interrupted");
			return null;
		} catch (ExecutionException e) {
			debug.println(1,
					"ThreadedSumKernel : Exception in execution, line unavailable.");
			e.printStackTrace();
			return null;
		}
		
		ThreadPoolServer.shutdownNow(te);
		
		return output;
	}

}
