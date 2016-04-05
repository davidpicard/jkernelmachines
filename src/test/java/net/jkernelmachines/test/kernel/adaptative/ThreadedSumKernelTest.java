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

    Copyright David Picard - 2014

*/
package net.jkernelmachines.test.kernel.adaptative;

import static org.junit.Assert.assertEquals;

import java.util.List;

import net.jkernelmachines.kernel.adaptative.ThreadedSumKernel;
import net.jkernelmachines.kernel.typed.index.IndexDoubleGaussL2;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.generators.GaussianGenerator;

import org.junit.Test;

/**
 * @author picard
 *
 */
public class ThreadedSumKernelTest {

	/**
	 * Test method for {@link net.jkernelmachines.kernel.adaptative.ThreadedSumKernel#getKernelMatrixLine(java.lang.Object, java.util.List)}.
	 */
	@Test
	public final void testGetKernelMatrixLine() {
		
		GaussianGenerator gg = new GaussianGenerator(16);
		List<TrainingSample<double[]>> l = gg.generateList(512);
		
		double[] w = new double[16];
		double sum = 0;
		for(int i = 0 ; i < 16 ; i++) {
			w[i] = Math.random();
			sum += w[i];
		}
		for(int i = 0 ; i < 16 ; i++) {
			w[i] /= sum;
		}
		
		ThreadedSumKernel<double[]> tsk = new ThreadedSumKernel<>();
		for(int i = 0 ; i < 16 ; i++) {
			tsk.addKernel(new IndexDoubleGaussL2(i), w[i]);
		}
		
		double[][] matrix = tsk.getKernelMatrix(l);
		double[][] lines = new double[l.size()][];
		for(int i = 0 ; i < l.size() ; i++) {
			lines[i] = tsk.getKernelMatrixLine(l.get(i).sample, l);
		}
		
		for(int i = 0 ; i < matrix.length ; i++) {
			for(int j = i ; j< matrix.length ; j++) {
				assertEquals(matrix[i][j], lines[i][j], 1e-10);
			}
		}
	}

}
