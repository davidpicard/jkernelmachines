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
package net.jkernelmachines.util;

import java.util.List;

import net.jkernelmachines.kernel.typed.DoubleLinear;
import net.jkernelmachines.type.TrainingSample;

/**
 * Util class for doing some preprocessing on data
 * 
 * @author picard
 *
 */
public class DataPreProcessing {

	/**
	 * Normalize a list of training samples of double[] to have l2-norm equal to
	 * 1
	 * 
	 * @param list
	 *            list
	 */
	public static void normalizeList(List<TrainingSample<double[]>> list) {
		if (list.isEmpty())
			return;
		DoubleLinear linear = new DoubleLinear();

		for (TrainingSample<double[]> t : list) {
			double[] desc = t.sample;
			double norm = Math.sqrt(linear.valueOf(desc, desc));
			if (norm > 0) {
				for (int x = 0; x < desc.length; x++)
					desc[x] /= norm;
			}
		}

	}

	/**
	 * Normalize a list of double[] to have l2-norm equal to 1
	 * 
	 * @param list
	 *            list
	 */
	public static void normalizeDoubleList(List<double[]> list) {
		if (list.isEmpty())
			return;
		DoubleLinear linear = new DoubleLinear();

		for (double[] desc : list) {
			double norm = Math.sqrt(linear.valueOf(desc, desc));
			for (int x = 0; x < desc.length; x++)
				desc[x] /= norm;
		}

	}

	/**
	 * Process a list of training samples of double[] to have 0 mean
	 * 
	 * @param list
	 *            list
	 */
	public static void centerList(List<TrainingSample<double[]>> list) {
		if (list.isEmpty())
			return;

		double[] mean = new double[list.get(0).sample.length];
		for (TrainingSample<double[]> t : list) {
			double[] d = t.sample;
			for (int x = 0; x < d.length; x++) {
				mean[x] += d[x];
			}
		}

		for (int x = 0; x < mean.length; x++)
			mean[x] /= list.size();

		for (TrainingSample<double[]> t : list) {
			double[] d = t.sample;
			for (int x = 0; x < d.length; x++) {
				d[x] -= mean[x];
			}
		}
	}

	/**
	 * Process a list of samples of double[] to have unit variance
	 * 
	 * @param list
	 *            list
	 */
	public static void reduceList(List<TrainingSample<double[]>> list) {
		if (list.isEmpty())
			return;

		double[] mean = new double[list.get(0).sample.length];
		double[] square = new double[mean.length];
		double[] factor = new double[square.length];
		for (TrainingSample<double[]> t : list) {
			double[] d = t.sample;
			for (int x = 0; x < d.length; x++) {
				mean[x] += d[x];
				square[x] += d[x] * d[x];
			}
		}

		for (int x = 0; x < mean.length; x++) {
			mean[x] /= list.size();
			square[x] /= list.size();
			factor[x] = Math.sqrt((square[x] - mean[x] * mean[x]));
			if (factor[x] == 0)
				factor[x] = 1.0;
		}

		for (TrainingSample<double[]> t : list) {
			double[] d = t.sample;
			for (int x = 0; x < d.length; x++) {
				d[x] /= factor[x];
			}
		}

	}
}
