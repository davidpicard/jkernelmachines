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
package net.jkernelmachines.io;

import java.io.FileReader;
import java.io.IOException;
import java.io.LineNumberReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import net.jkernelmachines.type.TrainingSample;

/**
 * IO helper to import Arff files.
 * 
 * @author picard
 *
 */
public class ArffImporter {
	
	/**
	 * <p>
	 * Read training samples from an Arff file.
	 * </p>
	 * <p>
	 * Currently, String, Date and relational data are not supported and simply ignored.
	 * Nominal data are converted to a n-dimensional real space where n is the length of the dictionary.
	 * The component corresponding to the nominal value is then set to 1, and the others to 0.
	 * The class attribute has to be called "Class", case insensitive.
	 * </p>
	 * @param filename
	 * @return list of training samples
	 * @throws IOException
	 */
	public static List<TrainingSample<double[]>> importFromFile(String filename) throws IOException {
		LineNumberReader lin = new LineNumberReader(new FileReader(filename));
		String line = null;
		
		HashMap<Integer, String[]> nominalMap = new HashMap<Integer, String[]>();
		List<Integer> skipIndices = new ArrayList<Integer>();
		
		int ind = 0;
		int dim = 0;
		int classIndex = -1;
		String[] classValues = new String[0];
		while((line = lin.readLine()) != null) {
			line = line.trim();
			// ignore comments and empty lines
			if(line.isEmpty() || line.startsWith("%")) {
				continue;
			}
			
			String[] toks = line.split("[ \t]+");
			for(int s = 0 ; s < toks.length ; s++) {
				toks[s] = toks[s].trim();
			}
			// dataset name
			if("@relation".equalsIgnoreCase(toks[0])) {
				// nothing
			}
			else if("@attribute".equalsIgnoreCase(toks[0])) {
				if(toks.length < 3) {
					continue;
				}
				
				if("numeric".equalsIgnoreCase(toks[2])) {
					ind++;
					dim++;
				}
				else if("real".equalsIgnoreCase(toks[2])) {
					ind++;
					dim++;
				}
				else if("integer".equalsIgnoreCase(toks[2])) {
					ind++;
					dim++;
				}
				else if("date".equalsIgnoreCase(toks[2])) {
					skipIndices.add(ind);
					ind++;
				}
				else if("string".equalsIgnoreCase(toks[2])) {
					skipIndices.add(ind);
					ind++;
				}
				else if(toks[2].startsWith("{") || (line.contains("{") && line.contains("}"))) {
					int b = line.indexOf('{')+1;
					int e = line.indexOf('}');
					String nom = line.substring(b, e).trim();
					String[] val = nom.split(",");
					for(int s = 0 ; s < val.length ; s++) {
						val[s] = val[s].trim();
					}
					
					// class or not
					if("class".equalsIgnoreCase(toks[1].replaceAll("[']+", "").trim())) {
						classIndex = ind;
						classValues = val;
						ind++;
					}
					else {
						nominalMap.put(ind, val);
						ind++;
						dim += val.length;	
					}
					
				}
				else {
					skipIndices.add(ind);
					ind++;
				}
			}
			else if("@data".equalsIgnoreCase(toks[0])) {
				break;
			}
		}
		
		List<TrainingSample<double[]>> list = new ArrayList<TrainingSample<double[]>>();
		while((line=lin.readLine())!=null) {
			// clean empty lines
			if(line.isEmpty()) {
				continue;
			}
			// clean comments
			if(line.startsWith("%")) {
				continue;
			}
			
			double[] v = new double[dim];
			int y = 0;
			String[] tok = line.split(",");
			int dimension = 0;
			
			for(int i = 0 ; i < ind ; i++) {
				// skip value
				if(skipIndices.contains(i)) {
					continue;
				}
				// sparse representation
				else if(tok[i].isEmpty()) {
					dimension++;
				}
				// missing attributes
				else if(tok[i].equals("?")) {
					dimension++;					
				}
				// case nominal
				else if(nominalMap.containsKey(i)) {
					String[] nom = nominalMap.get(i);
					for(int d = 0 ; d < nom.length ; d++) {
						if(nom[d].trim().equals(tok[i])) {
							v[dimension+d] = 1;
							break;
						}
					}
					dimension += nom.length;
				}
				// case class
				else if(classIndex == i) {
					if(classValues.length == 2) {
						if(tok[i].trim().equals(classValues[0])) {
							y = 1;
						}
						else {
							y = -1;
						}
					}
					else {
						for(int d = 0  ; d < classValues.length ; d++) {
							if(classValues[d].equals(tok[i])) {
								y = d+1;
							}
						}
					}
				}
				// case real
				else {
					v[dimension] = Double.parseDouble(tok[i]);
					dimension++;
				}
			}
			
			list.add(new TrainingSample<double[]>(v, y));
		}

		lin.close();
		return list;
	}
	
	public static void main(String[] args) throws Exception {
		importFromFile(args[0]);
	}

}
