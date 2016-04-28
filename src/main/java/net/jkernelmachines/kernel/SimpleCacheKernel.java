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

import java.util.List;

import net.jkernelmachines.type.TrainingSample;

/**
 * Very simple caching method for any kernel. Caches only the Gram matrix of a
 * specified list of training samples.
 * 
 * @author picard
 * 
 * @param <T> samples data type
 */
public final class SimpleCacheKernel<T> extends Kernel<T> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2417905029129394427L;

	final private Kernel<T> kernel;
	final private double matrix[][];

	/**
	 * Constructor using a kernel and a list of samples
	 * 
	 * @param k
	 *            the underlying of this caching kernel
	 * @param l
	 *            the list on which to compute the Gram matrix
	 */
	public SimpleCacheKernel(Kernel<T> k, List<TrainingSample<T>> l) {
		kernel = k;
		matrix = new ThreadedKernel<>(k).getKernelMatrix(l);
	}

	@Override
	final public double valueOf(T t1, T t2) {
		return kernel.valueOf(t1, t2);
	}

	@Override
	final public double valueOf(T t1) {
		return kernel.valueOf(t1);
	}

	@Override
	public double[][] getKernelMatrix(List<TrainingSample<T>> e) {

		return matrix;

	}

	/**
	 * Returns the underlying kernel
	 * 
	 * @return the cached kernel
	 */
	public Kernel<T> getKernel() {
		return kernel;
	}

	@Override
	public String toString() {
		return kernel.toString();
	}

}
