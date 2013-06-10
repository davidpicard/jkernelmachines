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
package fr.lip6.jkernelmachines.kernel.extra;

import fr.lip6.jkernelmachines.kernel.Kernel;

/**
 * <p>
 * Kernel with provided custom train and test matrices.
 * </p>
 * <p>
 * The datatype of input space is integer relative to row/column indices.
 * Therefore, the similarity between elements i and j is matrix[i][j]. <br />
 * Two matrices have to be provided: the first is train[nb_train][nb_train], and the second
 * is test[nb_train][nb_test].<br />
 * </p>
 * <p>
 * We suppose the train indices start at 0, and the test indices start at
 * nb_train (the number of rows in the train matrix). For example, if you have 5
 * training samples and 10 testing samples, the train matrix is 5x5 and the test
 * matrix is 5x10. This means that valid training indices are {0, 1, 2, 3, 4}
 * and valid testing indices are {5, 6, 7, 8, 9, 10, 11, 12, 13, 14}. <br />
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
