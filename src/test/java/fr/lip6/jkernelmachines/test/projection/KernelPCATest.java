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
package fr.lip6.jkernelmachines.test.projection;

import static org.junit.Assert.assertEquals;

import java.util.List;

import org.junit.Before;
import org.junit.Test;

import fr.lip6.jkernelmachines.kernel.typed.DoubleGaussL2;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;
import fr.lip6.jkernelmachines.projection.KernelPCA;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.generators.GaussianGenerator;

/**
 * @author picard
 *
 */
public class KernelPCATest {


	private List<TrainingSample<double[]>> list;
	private KernelPCA<double[]> pca;
	DoubleGaussL2 k;
	

	
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		
		int dim = 32;
		
		GaussianGenerator gen = new GaussianGenerator(dim, 0, 1.0);
		list = gen.generateList(dim);
		
		k = new DoubleGaussL2(2.0);
		pca = new KernelPCA<double[]>(k);
		
		pca.train(list);
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.projection.KernelPCA#projectList(java.util.List)}.
	 */
	@Test
	public final void testProjectListListOfTrainingSampleOfT() {
		double[][] m1 = k.getKernelMatrix(list);
		
		List<TrainingSample<double[]>> plist = pca.projectList(list);
		DoubleLinear lin = new DoubleLinear();
		double[][] m2 = lin.getKernelMatrix(plist);
// disable temporarily
/*		
		for(int i = 0 ; i < m1.length ; i++) {
			for(int j = i ; j < m1[0].length ; j++) {
				assertEquals(m1[i][j], m2[i][j]+pca.getMean(), 1e-10);
			}
		}
*/
	}

}
