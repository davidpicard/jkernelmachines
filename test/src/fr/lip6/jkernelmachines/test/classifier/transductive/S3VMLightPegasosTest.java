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
package fr.lip6.jkernelmachines.test.classifier.transductive;

import static org.junit.Assert.assertTrue;

import java.util.List;

import org.junit.Before;
import org.junit.Test;

import fr.lip6.jkernelmachines.classifier.transductive.S3VMLightPegasos;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.generators.GaussianGenerator;

/**
 * @author picard
 *
 */
public class S3VMLightPegasosTest {

	List<TrainingSample<double[]>> train;
	List<TrainingSample<double[]>> test;
	S3VMLightPegasos svm;
	
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		GaussianGenerator g = new GaussianGenerator(2, 10.0f, 1.0);
		train = g.generateList(10);
		test = g.generateList(10);

		svm = new S3VMLightPegasos();
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.classifier.transductive.S3VMLightPegasos#train(java.util.List, java.util.List)}.
	 */
	@Test
	public final void testTrain() {
		svm.train(train, test);
		for (TrainingSample<double[]> t : train) {
			double v = svm.valueOf(t.sample);
			assertTrue(t.label * v > 0);
		}

		for (TrainingSample<double[]> t : test) {
			double v = svm.valueOf(t.sample);
			assertTrue(t.label * v > 0);
		}
	}

}
