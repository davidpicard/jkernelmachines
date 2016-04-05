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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;

import net.jkernelmachines.type.TrainingSample;

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
		int line_index = 0;
		do {
			line = lin.readLine();
			if(line == null)
				break;

			StringTokenizer tokenizer = new StringTokenizer(line, "[ ]+");
			Map<Integer, Double> map = new HashMap<Integer, Double>();
			//make sure every vector is different by adding the line number
			map.put(-1, (double)++line_index);

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
				if(x < 1)
					continue;
				f[x-1] = m.get(x);
			}

			TrainingSample<double[]> t = new TrainingSample<double[]>(f, y);
			list.add(t);
		}

		lin.close();
		return list;

	}
}
