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

    Copyright David Picard - 2010

 */
package fr.lip6.jkernelmachines.kernel.adaptative;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

import fr.lip6.jkernelmachines.kernel.Kernel;
import fr.lip6.jkernelmachines.threading.ThreadPoolServer;
import fr.lip6.jkernelmachines.threading.ThreadedMatrixOperator;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.DebugPrinter;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

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
