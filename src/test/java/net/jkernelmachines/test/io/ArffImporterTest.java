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
package net.jkernelmachines.test.io;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.IOException;
import java.util.List;

import net.jkernelmachines.io.ArffImporter;
import net.jkernelmachines.type.TrainingSample;

import org.junit.Test;

/**
 * @author picard
 *
 */
public class ArffImporterTest {

	/**
	 * Test method for {@link net.jkernelmachines.io.ArffImporter#importFromFile(java.lang.String)}.
	 */
	@Test
	public final void testImportFromFile() {
		try {
			List<TrainingSample<double[]>> l = ArffImporter.importFromFile("resources/ionosphere.arff");
			assertEquals(351, l.size());
			assertEquals(34, l.get(0).sample.length);
			
		} catch (IOException e) {
			fail("Exception thrown: "+e.getMessage());
		}
		
	}

}
