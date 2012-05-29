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
package fr.lip6.io;

import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import fr.lip6.type.TrainingSample;

/**
 * Simple class to import data in csv format, with one sample per line:<br/>
 * attr1, attr2, ... , class
 * @author picard
 *
 */
public class CsvImporter {
	public static List<TrainingSample<double[]>> importFromFile(String filename)
			throws IOException {

		//the samples list
		List<TrainingSample<double[]>> list = new ArrayList<TrainingSample<double[]>>();
		
		LineNumberReader line = new LineNumberReader(new FileReader(filename));
		String l;
		//parse all lines
		while( (l = line.readLine()) != null) {
			
			String[] tok = l.split(",");
			double[] d = new double[tok.length - 1];
			
			// first n-1 fileds are attributes
			for(int i = 0 ; i < d.length ; i++)
				d[i] = Double.parseDouble(tok[i]);
			
			//last field is class
			int y = Integer.parseInt(tok[tok.length-1]);
			
			TrainingSample<double[]> t = new TrainingSample<double[]>(d, y);
			list.add(t);			
		}
		
		return list;

	}

}
