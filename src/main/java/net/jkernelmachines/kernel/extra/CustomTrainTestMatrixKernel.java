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
package net.jkernelmachines.kernel.extra;

import net.jkernelmachines.kernel.Kernel;

/**
 * <p>
 * Kernel with provided custom train and test matrices.
 * </p>
 * <p>
 * The datatype of input space is integer relative to row/column indices.
 * Therefore, the similarity between elements i and j is matrix[i][j].
 * Two matrices have to be provided: the first is train[nb_train][nb_train], and the second
 * is test[nb_train][nb_test].
 * </p>
 * <p>
 * We suppose the train indices start at 0, and the test indices start at
 * nb_train (the number of rows in the train matrix). For example, if you have 5
 * training samples and 10 testing samples, the train matrix is 5x5 and the test
 * matrix is 5x10. This means that valid training indices are {0, 1, 2, 3, 4}
 * and valid testing indices are {5, 6, 7, 8, 9, 10, 11, 12, 13, 14}. 
 * To get the similarity between the first training sample and the first testing
 * sample, you thus have to call valueOf(0, 5+0). The kernel automatically
 * translates the second index for the test matrix and returns test[0][0].
 * 
 * </p>
 * 
 * @author dpicard
 * 
 */
public class CustomTrainTestMatrixKernel extends Kernel<Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5379932592270965091L;
	private double train[][];
	private double test[][];

	private int nb_train;

	/**
	 * Constructor using the supplied Gram matrices.
	 * 
	 * @param train
	 *            the Gram matrix of underlying kernel function for train/train
	 *            similarity.
	 * @param test
	 *            the Gram matrix for the train/test similarity.
	 */
	public CustomTrainTestMatrixKernel(double train[][], double test[][]) {
		this.train = train;
		nb_train = train.length;
		this.test = test;
	}

	@Override
	public double valueOf(Integer t1, Integer t2) {
		int min, max;
		if (t1 <= t2) {
			min = t1;
			max = t2;
		} else {
			max = t1;
			min = t2;
		}

		// both in test
		if (min > nb_train) {
			return 0;
		}

		// second in train
		if (max < nb_train) {
			return train[min][max];
		}
		// second in test
		else {
			return test[min][max - nb_train];
		}
	}

	@Override
	public double valueOf(Integer t1) {
		if (t1 > nb_train) {
			return 0.;
		}
		return train[t1][t1];
	}

}
