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
package fr.lip6.jkernelmachines.io;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * @author picard
 *
 */
public class FvecImporter {
	
	private byte buf[];
	private DataInputStream input;
	private DataOutputStream output;
	
	/**
	 * Reads a file in fvec (INRIA) format and return the samples as a list of double arrays.
	 * @param filename to file to read
	 * @return a list of the samples as double arrays
	 * @throws IOException
	 */
	public List<double[]> readFile(String filename) throws IOException {
		
		File f = new File(filename);
		input = new DataInputStream(new FileInputStream(f));
		buf = new byte[4];
		
		// read first feature
		int dim = readInt();
		double[] d = new double[dim];
		for(int x = 0 ; x < dim ; x++)
			d[x] = readFloat();
		
		long size = f.length();
		if( size%(4+4*dim) != 0)
			throw new EOFException("Wrong file size, not matching dimension "+dim);
		
		List<double[]> list = new ArrayList<double[]>();
		list.add(d);
		
		long nb_samples = (size-4)/(4*dim);
		for(long i = 1 ; i < nb_samples; i++) {
			readInt();
			d = new double[dim];
			for(int x = 0 ; x < dim ; x++)
				d[x] = readFloat();
			
			list.add(d);
		}
		
		return list;
	}
	
	/* little endian */
	private int readInt() throws IOException {
		int r = 0;
		
		if(buf == null) {
			return -1;
		}
		
		if(input == null) {
			return -1;
		}
		
		input.readFully(buf);
		
		r = (buf[3] << 24) | ((buf[2] & 0xFF) << 16) | ((buf[1] & 0xFF) << 8)  | (buf[0] & 0xFF);		
		return r;
	}
	
	/* little endian, thus using readInt() */
	private float readFloat() throws IOException {
		return Float.intBitsToFloat(readInt());
	}

}
