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
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Class providing routines to import data i fvec format (usefull in computer
 * vision).
 * 
 * @author picard
 * 
 */
public class FvecImporter {

	private byte buf[];
	private DataInputStream input;
	private DataOutputStream output;

	/**
	 * Reads a file in fvec (INRIA) format and return the samples as a list of
	 * double arrays.
	 * 
	 * @param filename
	 *            to file to read
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
		for (int x = 0; x < dim; x++)
			d[x] = readFloat();

		long size = f.length();
		if (size % (4 + 4 * dim) != 0)
			throw new EOFException("Wrong file size, not matching dimension "
					+ dim);

		List<double[]> list = new ArrayList<double[]>();
		list.add(d);

		long nb_samples = (size) / (4 * dim + 4);
		for (long i = 1; i < nb_samples; i++) {
			readInt();
			d = new double[dim];
			for (int x = 0; x < dim; x++)
				d[x] = readFloat();

			list.add(d);
		}

		return list;
	}
	
	/**
	 * Reads a stream in fvec (INRIA) format and return the samples as a list of
	 * double arrays.
	 * 
	 * @param i inputstream to read data from
	 * @return a list of the samples as double arrays
	 * @throws IOException
	 */
	public List<double[]> readInputStream(InputStream i) throws IOException {
		input = new DataInputStream(i);
		buf = new byte[4];

		// read first feature
		int dim = readInt();
		double[] d = new double[dim];
		for (int x = 0; x < dim; x++)
			d[x] = readFloat();

		List<double[]> list = new ArrayList<double[]>();
		list.add(d);

		while (true) {
			try {
				readInt();
				d = new double[dim];
				for (int x = 0; x < dim; x++) {
					d[x] = readFloat();
				}
				list.add(d);
			} catch (IOException ioe) {
				// end of stream
				break;
			}
		}

		return list;
	}

	/**
	 * Writes a list of features (double arrays) to a file in fvec (INRIA)
	 * format.
	 * 
	 * @param filename
	 *            the name of the file to be written
	 * @param list
	 *            the list of features
	 * @throws IOException
	 */
	public void writeFile(String filename, List<double[]> list)
			throws IOException {
		buf = new byte[4];

		File f = new File(filename);
		output = new DataOutputStream(new FileOutputStream(f));

		for (double[] d : list) {
			writeInt(d.length);
			for (int x = 0; x < d.length; x++) {
				writeFloat((float) d[x]);
			}
		}

	}

	/* little endian */
	private int readInt() throws IOException {
		int r = 0;

		if (buf == null) {
			throw new NullPointerException("buf is null");
		}

		if (input == null) {
			throw new NullPointerException("input is not set");
		}

		input.readFully(buf);

		r = (buf[3] << 24) | ((buf[2] & 0xFF) << 16) | ((buf[1] & 0xFF) << 8)
				| (buf[0] & 0xFF);
		return r;
	}

	/* write int in little endian */
	private void writeInt(int i) throws IOException {
		if (buf == null) {
			throw new NullPointerException("buf is null");
		}
		if (output == null) {
			throw new NullPointerException("output is not set");
		}

		buf[0] = (byte) (i & 0xFF);
		buf[1] = (byte) ((i >> 8) & 0xFF);
		buf[2] = (byte) ((i >> 16) & 0xFF);
		buf[3] = (byte) (i >> 24);

		output.write(buf);

	}

	/* little endian, thus using readInt() */
	private float readFloat() throws IOException {
		return Float.intBitsToFloat(readInt());
	}

	private void writeFloat(float f) throws IOException {
		writeInt(Float.floatToIntBits(f));
	}

}
