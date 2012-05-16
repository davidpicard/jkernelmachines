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

    Copyright David Picard - 2010

*/
package fr.lip6.test.kernel;

import java.util.ArrayList;
import java.util.List;

import fr.lip6.kernel.typed.DoubleLinear;

/**
 * Simple example of the Linear Kernel on vectors of doubles.
 * @author picard
 *
 */
public class TestDoubleLinear {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		int nb = 100, dim = 1000000;
		double sparse = 0.1;
		DoubleLinear linear = new DoubleLinear();
		
		List<double[]> list = new ArrayList<double[]>();
		
		for(int i = 0 ; i < nb ; i++) {
			double[] x = new double[dim];
			for(int j = 0 ; j < dim ; j++) {
				if(Math.random() < sparse)
					x[j] = Math.random();
			}
			list.add(x);
		}
		
		System.out.println("generated.");
		double sum = 0;
		long tim = System.currentTimeMillis();
		double[] x0 = list.get(0);
		for(int i = 0 ; i < nb ; i++) {
			double[] x = list.get(i);
				sum += linear.valueOf(x0, x);
		}
		long done = System.currentTimeMillis() - tim;
		System.out.println("sum : "+sum+" done in "+done+"ms");

	}

}
