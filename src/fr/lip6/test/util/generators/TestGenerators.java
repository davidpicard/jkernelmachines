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

    Copyright David Picard - 2012

 */
package fr.lip6.test.util.generators;

import java.util.List;

import fr.lip6.type.TrainingSample;
import fr.lip6.util.DebugPrinter;
import fr.lip6.util.generators.GaussianGenerator;

/**
 * Test cases for generator classes
 * @author picard
 * 
 */
public class TestGenerators {

	/**
	 * @param args ignored
	 */
	public static void main(String[] args) {

		DebugPrinter debug = new DebugPrinter();
		int good = 0;

		if (testGaussianGenerator()) {
			good++;
		} else {
			debug.println(0, "Warning: GaussIanGenerator failed.");
		}

		debug.println(0, "Testing generators: " + good + "/1 test validated.");

	}

	private static boolean testGaussianGenerator() {

		GaussianGenerator gg = new GaussianGenerator();
		// generate 100 samples
		List<TrainingSample<double[]>> list = gg.generateList(100);
		// test number
		if (list.size() != 100)
			return false;

		// test number of positives and negatives
		int p = 0, n = 0;
		for (TrainingSample<double[]> t : list) {
			if (t.label > 0)
				p++;
			else
				n++;
		}
		if (p != 50 || n != 50)
			return false;

		// generate 100 samples
		list = gg.generateList(10, 90);
		// test number
		if (list.size() != 100)
			return false;

		// test number of positives and negatives
		p = 0;
		n = 0;
		for (TrainingSample<double[]> t : list) {
			if (t.label > 0)
				p++;
			else
				n++;
		}
		if (p != 10 || n != 90)
			return false;

		return true;
	}
}
