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
package net.jkernelmachines.test.density;

import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;

import net.jkernelmachines.density.DoubleGaussianMixtureModel;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.generators.GaussianGenerator;

import org.junit.Before;
import org.junit.Test;

/**
 * Tast case for the DoubleGaussianMixtureModel class.
 * 
 * @author picard
 *
 */
public class DoubleGaussianMixtureModelTest {

	List<double[]> train;
	
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		GaussianGenerator gen = new GaussianGenerator(8, 2, 0.1);
		List<TrainingSample<double[]>> list = gen.generateList(1000, 1000);
		train = new ArrayList<double[]>();
		for(TrainingSample<double[]> t : list) {
			train.add(t.sample);
		}
	}

	/**
	 * Test method for {@link net.jkernelmachines.density.DoubleGaussianMixtureModel#train(java.util.List)}.
	 */
	@Test
	public final void testTrainListOfdouble() {
		DoubleGaussianMixtureModel gmm = new DoubleGaussianMixtureModel(2);
		gmm.train(train);
		
		for(double[] x : train) {
			assertTrue(gmm.valueOf(x) > 0);
			assertTrue(gmm.valueOf(x) <= 1);
		}
	}

}
