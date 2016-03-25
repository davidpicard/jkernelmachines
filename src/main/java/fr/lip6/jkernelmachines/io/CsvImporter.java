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
package fr.lip6.jkernelmachines.io;

import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.List;

import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * Simple class to import data in csv format, with one sample per line:
 * attr1, attr2, ... , class
 * 
 * position of the class label can be arbitrary
 * 
 * @author picard
 * 
 */
public class CsvImporter {
	/**
	 * Importer with full settings.
	 * @param filename the file containing the data
	 * @param sep the token which separates the values
	 * @param labelPosition the position of the class label
	 * @return the full list of TrainingSample
	 * @throws IOException
	 */
	public static List<TrainingSample<double[]>> importFromFile(
			String filename, String sep, int labelPosition) throws IOException {

		// the samples list
		List<TrainingSample<double[]>> list = new ArrayList<TrainingSample<double[]>>();

		LineNumberReader line = new LineNumberReader(new FileReader(filename));
		String l;
		// parse all lines
		while ((l = line.readLine()) != null) {

			String[] tok = l.split(sep);
			double[] d = new double[tok.length - 1];
			int y = 0;

			if(labelPosition == -1) {
			// first n-1 fields are attributes
			for (int i = 0; i < d.length; i++)
				d[i] = Double.parseDouble(tok[i]);
				// last field is class
				y = Integer.parseInt(tok[tok.length - 1]);

			}
			else if(labelPosition < d.length){
				for(int i = 0 ; i < labelPosition ; i++)
					d[i] = Double.parseDouble(tok[i]);
				for(int i = labelPosition+1 ; i < d.length ; i++)
					d[i-1] = Double.parseDouble(tok[i]);
				y = Integer.parseInt(tok[labelPosition]);
			}

			
			TrainingSample<double[]> t = new TrainingSample<double[]>(d, y);
			list.add(t);
		}

		line.close();
		return list;

	}

	/**
	 * CSV import routine with delimiter set to ","
	 * @param filename
	 * @param labelPosition
	 * @return The list of training samples
	 * @throws IOException
	 */
	public static List<TrainingSample<double[]>> importFromFile(
			String filename, int labelPosition) throws IOException {
		return importFromFile(filename, ",", labelPosition);
	}

	/**
	 * CSV import routine with label position set to the last value
	 * @param filename
	 * @param sep
	 * @return The list of training samples
	 * @throws IOException
	 */
	public static List<TrainingSample<double[]>> importFromFile(
			String filename, String sep) throws IOException {
		return importFromFile(filename, sep, -1);
	}

	/**
	 * CSV import routine with default parameters (separator is "," and the label is the last value)
	 * @param filename the file containing the values
	 * @return The list of training samples
	 * @throws IOException
	 */
	public static List<TrainingSample<double[]>> importFromFile(String filename)
			throws IOException {
		return importFromFile(filename, ",", -1);
	}

}
