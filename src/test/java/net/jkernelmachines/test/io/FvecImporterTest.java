/*******************************************************************************
 * Copyright (c) 2016, David Picard.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************/
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
