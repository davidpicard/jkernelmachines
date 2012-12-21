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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 *  Simple class to import data in libsvm format.
 * @author picard
 *
 */
public class LibSVMImporter {

	public static List<TrainingSample<double[]>> importFromFile(String filename)
			throws IOException {

		Map<Map<Integer, Double>, Integer> features = new HashMap<Map<Integer, Double>, Integer>();

		LineNumberReader lin = new LineNumberReader(new FileReader(filename));
		String line = null;
		int max_attr = 0;
		do {
			line = lin.readLine();
			if(line == null)
				break;

			StringTokenizer tokenizer = new StringTokenizer(line, "[ ]+");
			Map<Integer, Double> map = new HashMap<Integer, Double>();

			// class attribute
			int y = Integer.parseInt(tokenizer.nextToken());

			// read atributes
			while (tokenizer.hasMoreTokens()) {
				String s[] = tokenizer.nextToken().split(":");
				int pos = Integer.parseInt(s[0]);
				double val = Double.parseDouble(s[1]);

				// update max attr
				if (pos > max_attr)
					max_attr = pos;

				// add to map
				map.put(pos, val);

			}

			// add to features
			features.put(map, y);

		} while (line != null);

		// convert map to lists and arrays
		List<TrainingSample<double[]>> list = new ArrayList<TrainingSample<double[]>>();
		for (Map<Integer, Double> m : features.keySet()) {
			int y = features.get(m);

			double[] f = new double[max_attr];
			for (int x : m.keySet()) {
				f[x-1] = m.get(x);
			}

			TrainingSample<double[]> t = new TrainingSample<double[]>(f, y);
			list.add(t);
		}

		lin.close();
		return list;

	}
}
