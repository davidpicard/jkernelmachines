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
package fr.lip6.jkernelmachines.test.classifier;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.List;

import org.junit.Before;
import org.junit.Test;

import fr.lip6.jkernelmachines.classifier.DoubleSGD;
import fr.lip6.jkernelmachines.classifier.DoubleSGDQN;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.generators.GaussianGenerator;

/**
 * @author picard
 *
 */
public class DoubleSGDTest {

	
	List<TrainingSample<double[]>> train;
	DoubleSGD svm;
	
	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		GaussianGenerator g = new GaussianGenerator(10, 5.0f, 1.0);
		train = g.generateList(10);
		
		svm = new DoubleSGD();
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.classifier.DoubleSGD#train(java.util.List)}.
	 */
	@Test
	public final void testTrainListOfTrainingSampleOfdouble() {
		svm.train(train);
		for(TrainingSample<double[]> t : train) {
			double v = t.label * svm.valueOf(t.sample);
			assertTrue(v > 0);
		}
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.classifier.DoubleSGD#trainOnce(java.util.List)}.
	 */
	@Test
	public final void testTrainOnce() {
		svm.train(train);
		svm.trainOnce(train);
		for(TrainingSample<double[]> t : train) {
			double v = t.label * svm.valueOf(t.sample);
			assertTrue(v > 0);
		}
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.classifier.DoubleSGD#setLoss(int)}.
	 */
	@Test
	public final void testSetLoss() {
		svm.setLoss(DoubleSGD.HINGELOSS);
		assertEquals(DoubleSGD.HINGELOSS, svm.getLoss());
		
		svm.setLoss(DoubleSGD.LOGLOSS);
		assertEquals(DoubleSGD.LOGLOSS, svm.getLoss());
		
		svm.setLoss(DoubleSGD.LOGLOSSMARGIN);
		assertEquals(DoubleSGD.LOGLOSSMARGIN, svm.getLoss());
		
		svm.setLoss(DoubleSGD.SMOOTHHINGELOSS);
		assertEquals(DoubleSGD.SMOOTHHINGELOSS, svm.getLoss());
		
		svm.setLoss(DoubleSGD.SQUAREDHINGELOSS);
		assertEquals(DoubleSGD.SQUAREDHINGELOSS, svm.getLoss());
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.classifier.DoubleSGD#setLambda(double)}.
	 */
	@Test
	public final void testSetLambda() {
		svm.setLambda(1e-3);
		assertEquals(1e-3, svm.getLambda(), 1e-15);
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.classifier.DoubleSGD#setHasBias(boolean)}.
	 */
	@Test
	public final void testSetHasBias() {
		svm.setHasBias(true);
		assertTrue(svm.isHasBias());
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.classifier.DoubleSGD#setEpochs(int)}.
	 */
	@Test
	public final void testSetEpochs() {
		svm.setEpochs(10);
		assertEquals(10, svm.getEpochs());
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.classifier.DoubleSGD#setShuffle(boolean)}.
	 */
	@Test
	public final void testSetShuffle() {
		svm.setShuffle(true);
		assertTrue(svm.isShuffle());
	}

}
