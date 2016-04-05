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
package net.jkernelmachines.test.io;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import net.jkernelmachines.io.FvecImporter;

import org.junit.Test;

/**
 * @author picard
 *
 */
public class FvecImporterTest {

	/**
	 * Test method for {@link net.jkernelmachines.io.FvecImporter#readFile(java.lang.String)}.
	 */
	@Test
	public final void testReadFile() {
		try {
			FvecImporter fvecimp = new FvecImporter();
			List<double[]> l = fvecimp.readFile("resources/dict.fvec");
			assertEquals(32, l.size());
			assertEquals(40, l.get(0).length);
		}
		catch(IOException e) {

			fail("Exception thrown: "+e.getMessage());
		}
	}
	
	/**
	 * Test method for {@link fr.lip6.jkernelmachines.io.FvecImporter#writeFile(java.lang.String, java.util.List<double[]>)}.
	 */
	@Test
	public final void testWriteFile() {
		try {
			FvecImporter fvecimp = new FvecImporter();
			
			List<double[]> l = new ArrayList<double[]>();
			double[] d = new double[10];
			for(int i = 0 ; i < 10 ; i++)
				d[i] = i;
			l.add(d);
			
			fvecimp.writeFile("resources/testwrite.fvec", l);
			
			File f = new File("resources/testwrite.fvec");
			
			assertEquals(4+4*10, f.length());
			
			
			l = fvecimp.readFile("resources/testwrite.fvec");
			assertEquals(10, l.get(0).length);
			for(int i = 0 ; i < 10 ; i++)
				assertEquals(i, l.get(0)[i], 1e-7);
			
			f.delete();
		}
		catch(IOException e) {

			fail("Exception thrown: "+e.getMessage());
		}
	}

}
