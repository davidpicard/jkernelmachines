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

import fr.lip6.jkernelmachines.classifier.GradMKL;
import fr.lip6.jkernelmachines.classifier.LaSVM;
import fr.lip6.jkernelmachines.classifier.SimpleMKL;
import fr.lip6.jkernelmachines.kernel.typed.DoubleGaussL2;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.generators.GaussianGenerator;

/**
 * @author picard
 *
 */
public class SimpleMKLTest {

	List<TrainingSample<double[]>> train;
	SimpleMKL<double[]> svm;

	/**
	 * @throws java.lang.Exception
	 */
	@Before
	public void setUp() throws Exception {
		
		GaussianGenerator g = new GaussianGenerator(10, 5.0f, 1.0);
		train = g.generateList(10);
		
		svm = new SimpleMKL<double[]>();
		svm.addKernel(new DoubleGaussL2());
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.classifier.SimpleMKL#addKernel(fr.lip6.jkernelmachines.kernel.Kernel)}.
	 */
	@Test
	public final void testAddKernel() {
		DoubleGaussL2 k = new DoubleGaussL2(1.0);
		svm.addKernel(k);
		assertTrue(svm.getKernels().contains(k));
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.classifier.SimpleMKL#train(java.util.List)}.
	 */
	@Test
	public final void testTrainListOfTrainingSampleOfT() {
		svm.train(train);
		for(TrainingSample<double[]> t : train) {
			double v = t.label * svm.valueOf(t.sample);
			assertTrue(v > 0);
		}
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.classifier.SimpleMKL#setC(double)}.
	 */
	@Test
	public final void testSetC() {
		svm.setC(10.0);
		assertEquals(10.0, svm.getC(), 1e-15);
	}

	/**
	 * Test method for {@link fr.lip6.jkernelmachines.classifier.SimpleMKL#setClassifier(fr.lip6.jkernelmachines.classifier.KernelSVM)}.
	 */
	@Test
	public final void testSetClassifier() {
		LaSVM<double[]> lasvm = new LaSVM<double[]>(new DoubleGaussL2());
		svm.setClassifier(lasvm);
		assertEquals(lasvm, svm.getClassifier());
	}

}
