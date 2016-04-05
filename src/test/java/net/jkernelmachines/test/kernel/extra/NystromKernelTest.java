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
package net.jkernelmachines.test.kernel.extra;

import static org.junit.Assert.assertEquals;

import java.util.List;

import net.jkernelmachines.kernel.extra.NystromKernel;
import net.jkernelmachines.kernel.typed.DoubleGaussL2;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.generators.GaussianGenerator;

import org.junit.Before;
import org.junit.Test;

/**
 * Test case for the NystromKernel class
 * @author picard
 *
 */
public class NystromKernelTest {
	
	List<TrainingSample<double[]>> list;
	DoubleGaussL2 dk;
	NystromKernel<double[]> nk;
	int nb_samples = 50;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		GaussianGenerator g = new GaussianGenerator(2, 1, 0.5);
		list = g.generateList(nb_samples);
		dk = new DoubleGaussL2();
		nk = new NystromKernel<double[]>(dk);
	}

	/**
	 * Test method for {@link net.jkernelmachines.kernel.extra.NystromKernel#train(java.util.List)}.
	 */
	@Test
	public final void testTrain() {
		double[][] matrix = dk.getKernelMatrix(list);
		nk.train(list);
		double[][] nk_matrix = nk.getKernelMatrix(list);
		
		for(int i = 0 ; i < matrix.length ; i++) {
			for(int j = i ; j< matrix.length ; j++) {
				assertEquals(matrix[i][j], nk_matrix[i][j], 1e-10);
			}
		}
	}

	/**
	 * Test method for {@link net.jkernelmachines.kernel.extra.NystromKernel#activeTrain(java.util.List, int, int, int)}.
	 */
	@Test
	public final void testActiveTrain() {
		double[][] matrix = dk.getKernelMatrix(list);
		nk.activeTrain(list, nb_samples, 1, 5);
		double[][] nk_matrix = nk.getKernelMatrix(list);
		
		for(int i = 0 ; i < matrix.length ; i++) {
			for(int j = i ; j< matrix.length ; j++) {
				assertEquals(matrix[i][j], nk_matrix[i][j], 1e-10);
			}
		}
	}

}
