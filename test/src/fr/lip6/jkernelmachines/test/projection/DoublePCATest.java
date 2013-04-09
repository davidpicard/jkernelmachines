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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.Before;
import org.junit.Test;

import fr.lip6.jkernelmachines.projection.DoublePCA;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * Test methods for PCA
 * 
 * @author picard
 *
 */
public class DoublePCATest {
	
	private List<TrainingSample<double[]>> list;
	private DoublePCA pca;
	
	private int nbSamples = 20;
	private int dim = 5;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		
		list = new ArrayList<TrainingSample<double[]>>();
		
		for(int i = 0 ; i < nbSamples ; i++) {
			double[] x = new double[dim];
			for(int d = 0 ; d < dim ; d++) {
				x[d] = Math.random();
			}
			
			list.add(new TrainingSample<double[]>(x, 1));
		}
		
		pca = new DoublePCA();
		pca.train(list);
		
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.projection.DoublePCA#projectList(java.util.List)}.
	 */
	@Test
	public final void testProjectListListOfTrainingSampleOfdouble() {
		
		List<TrainingSample<double[]>> proj = pca.projectList(list);
		
		// compute cov
		double[][] cov = new double[dim][dim];
		for(int i = 0 ; i < dim ; i++) {
			for(int j = 0 ; j < dim ; j++) {
				for(TrainingSample<double[]> t : proj) {
					cov[i][j] += t.sample[i] * t.sample[j];
				}
				
				cov[i][j] /= list.size();
				if(i!=j) {
					assertEquals(0, cov[i][j], 1e-8);
				}
			}
		}
		
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.projection.DoublePCA#projectList(java.util.List, boolean)}.
	 */
	@Test
	public final void testProjectListListOfTrainingSampleOfdoubleBoolean() {
		
		List<TrainingSample<double[]>> proj = pca.projectList(list, true);
		
		// compute cov
		double[][] cov = new double[dim][dim];
		for(int i = 0 ; i < dim ; i++) {
			for(int j = 0 ; j < dim ; j++) {
				for(TrainingSample<double[]> t : proj) {
					cov[i][j] += t.sample[i] * t.sample[j];
				}
				cov[i][j] /= list.size();
				if(i!=j) {
					assertEquals(0, cov[i][j], 1e-8);
				}
				else {
					assertEquals(1, cov[i][j], 1e-8);
				}
			}
		}
	}

}
