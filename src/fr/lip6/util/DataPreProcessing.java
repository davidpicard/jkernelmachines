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
package fr.lip6.util;

import java.util.List;

import fr.lip6.kernel.typed.DoubleLinear;
import fr.lip6.type.TrainingSample;

/**
 * Util class for doing some preprocessing on data
 * @author picard
 *
 */
public class DataPreProcessing {
	
	/**
	 * Normalize a list of training samples of double[] to have l2-norm equal to 1
	 * @param list
	 */
	public static void normalizeList(List<TrainingSample<double[]>> list) {
		if(list.isEmpty())
			return;
		DoubleLinear linear = new DoubleLinear();
		
		for(TrainingSample<double[]> t : list) {
			double[] desc = t.sample;
			double norm = Math.sqrt(linear.valueOf(desc, desc));
			for(int x = 0 ; x < desc.length ; x++)
				desc[x] /= norm;
		}
		
	}

	/**
	 * Process a list of training samples of double[] to have 0 mean
	 * @param list
	 */
	public static void centerList(List<TrainingSample<double[]>> list) {
		if(list.isEmpty())
			return;
		
		double[] mean = new double[list.get(0).sample.length];
		for(TrainingSample<double[]> t : list) {
			double[] d = t.sample;
			for(int x = 0 ; x < d.length ; x++) {
				mean[x] += d[x];
			}
		}

		for(int x = 0 ; x < mean.length ; x++)
			mean[x] /= list.size();
		
		for(TrainingSample<double[]> t : list) {
			double[] d = t.sample;
			for(int x = 0 ; x < d.length ; x++) {
				d[x] -= mean[x];
			}
		}
	}
	
	/**
	 * Process a list of samples of double[] to have unit variance
	 * @param list
	 */
	public static void reduceList(List<TrainingSample<double[]>> list) {
		if(list.isEmpty())
			return;
		
		double[] mean = new double[list.get(0).sample.length];
		double[] square = new double[mean.length];
		double[] factor = new double[square.length];
		for(TrainingSample<double[]> t : list) {
			double[] d = t.sample;
			for(int x = 0 ; x < d.length ; x++) {
				mean[x] += d[x];
				square[x] += d[x]*d[x];
			}
		}

		for(int x = 0 ; x < mean.length ; x++) {
			mean[x] /= list.size();
			square[x] /= list.size();
			factor[x] = Math.sqrt((square[x] - mean[x]*mean[x]));
			if(factor[x] == 0)
				factor[x] = 1.0;
		}
		
		for(TrainingSample<double[]> t : list) {
			double[] d = t.sample;
			for(int x = 0 ; x < d.length ; x++) {
				d[x] /= factor[x];
			}
		}
				
		
		
	}
}
