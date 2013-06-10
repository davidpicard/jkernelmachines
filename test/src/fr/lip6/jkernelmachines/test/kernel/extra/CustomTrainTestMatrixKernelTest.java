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
package fr.lip6.jkernelmachines.test.kernel.extra;

import static org.junit.Assert.assertEquals;

import java.util.List;

import org.junit.Test;

import fr.lip6.jkernelmachines.kernel.extra.CustomTrainTestMatrixKernel;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.generators.GaussianGenerator;

/**
 * @author picard
 *
 */
public class CustomTrainTestMatrixKernelTest {

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.kernel.extra.CustomTrainTestMatrixKernel#valueOf(java.lang.Integer, java.lang.Integer)}.
	 */
	@Test
	public final void testValueOfIntegerInteger() {
		
		// generate data
		GaussianGenerator gen = new GaussianGenerator(2);
		List<TrainingSample<double[]>> list = gen.generateList(5, 10);
		
		DoubleLinear lin = new DoubleLinear();
		double[][] matrix = lin.getKernelMatrix(list);
		
		double[][] train = new double[5][5];
		double[][] test = new double[10][10];
		
		// fill train
		for(int i = 0 ; i < 5 ; i++) {
			for(int j = 0 ; j < 5 ; j++) {
				train[i][j] = matrix[i][j];
			}
		}
		// fill test
		for(int i = 0 ; i < 5 ; i++) {
			for(int j = 0 ; j < 10 ; j++) {
				test[i][j] = matrix[i][5+j];
			}
		}
		
		// build custom kernel
		CustomTrainTestMatrixKernel kernel = new CustomTrainTestMatrixKernel(train, test);
		
		// assert values from train to end of test
		for(int i = 0 ; i < 5 ; i++) {
			for(int j = 0 ; j < 15 ; j++) {
				assertEquals(matrix[i][j], kernel.valueOf(i, j), 1e-15);
				assertEquals(matrix[j][i], kernel.valueOf(j, i), 1e-15);
			}
		}
		
	}

}
