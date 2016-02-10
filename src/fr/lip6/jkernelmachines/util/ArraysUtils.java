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
package fr.lip6.jkernelmachines.util;

/**
 * @author picard
 *
 */
public class ArraysUtils {

	/**
	 * computes the mean of a double array
	 * @param a the array
	 * @return the mean
	 */
	public static double mean(double[] a) {
		double sum = 0;
		for(int i = 0 ; i < a.length ; i++) {
			sum += a[i];
		}
		return sum/(double)a.length;
	}
	
	/**
	 * Computes the standard deviation of an array
	 * @param a the array
	 * @return the standard deviation
	 */
	public static double stddev(double[] a) {
		double std = 0;
		double ave = mean(a);
		
		for(double d : a)
			std += (d-ave)*(d-ave);
		
		return Math.sqrt(std/(double)a.length);
	}
}
