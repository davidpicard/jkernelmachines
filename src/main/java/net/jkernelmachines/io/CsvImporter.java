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
package net.jkernelmachines.io;

import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.List;

import net.jkernelmachines.type.TrainingSample;

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
	 * @throws IOException if file cannot be opened
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
	 * @param filename filename
	 * @param labelPosition position
	 * @return The list of training samples
	 * @throws IOException if file cannot be opened
	 */
	public static List<TrainingSample<double[]>> importFromFile(
			String filename, int labelPosition) throws IOException {
		return importFromFile(filename, ",", labelPosition);
	}

	/**
	 * CSV import routine with label position set to the last value
	 * @param filename filename
	 * @param sep separator
	 * @return The list of training samples
	 * @throws IOException if file cannot be opened
	 */
	public static List<TrainingSample<double[]>> importFromFile(
			String filename, String sep) throws IOException {
		return importFromFile(filename, sep, -1);
	}

	/**
	 * CSV import routine with default parameters (separator is "," and the label is the last value)
	 * @param filename the file containing the values
	 * @return The list of training samples
	 * @throws IOException if file cannot be opened
	 */
	public static List<TrainingSample<double[]>> importFromFile(String filename)
			throws IOException {
		return importFromFile(filename, ",", -1);
	}

}
