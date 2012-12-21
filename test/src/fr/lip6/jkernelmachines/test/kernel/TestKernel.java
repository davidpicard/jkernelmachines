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
package fr.lip6.jkernelmachines.test.kernel;

import fr.lip6.jkernelmachines.kernel.typed.DoubleGaussChi2;
import fr.lip6.jkernelmachines.kernel.typed.DoubleGaussL2;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;
import fr.lip6.jkernelmachines.util.DebugPrinter;

/**
 * Test cases for some kernel classes
 * 
 * @author picard
 *
 */
public class TestKernel {
	
	static DebugPrinter debug = new DebugPrinter();

	/**
	 * @param args ignored
	 */
	public static void main(String[] args) {
		
		// two testing samples
		double first[] = {1, 0};
		double second[] = {0, 1};
		
		DebugPrinter.setDebugLevel(0);
		
		int good = 0;
		if(testDoubleLinear(first, second))
			good++;
		else {
			debug.println(0, "Warning TestDoubleLinear failed!");
		}
		if(testDoubleGaussL2(first, second))
			good++;
		else {
			debug.println(0, "Warning TestDoubleGaussL2 failed!");
		}
		if(testDoubleGaussChi2(first, second))
			good++;
		else {
			debug.println(0, "Warning TestDoubleGaussChi2 failed!");
		}
		
		debug.println(0, "Testing kernels: "+good+"/3 validated.");

	}

	private static boolean testDoubleLinear(double fir[], double sec[]) {
		DoubleLinear kernel = new DoubleLinear();
		
		if(kernel.valueOf(fir, fir) != 1.0) {
			debug.println(0, "TestDoubleLinear: similarity to self failed!");
			return false;
		}
		
		if(kernel.valueOf(fir, sec) != 0.) {
			debug.println(0, "TestDoubleLinear: orthogonal samples similarity failed!");
			return false;
		}
		
		return true;
	}
	
	

	private static boolean testDoubleGaussL2(double fir[], double sec[]) {
		DoubleGaussL2 kernel = new DoubleGaussL2();
		
		if(kernel.valueOf(fir, fir) != 1.0) {
			debug.println(0, "testDoubleGaussL2: similarity to self failed!");
			return false;
		}
		
		if(kernel.valueOf(fir, sec) >= 1.) {
			debug.println(0, "testDoubleGaussL2: orthogonal samples similarity failed!");
			return false;
		}
		
		return true;
	}


	private static boolean testDoubleGaussChi2(double fir[], double sec[]) {
		DoubleGaussChi2 kernel = new DoubleGaussChi2();
		
		if(kernel.valueOf(fir, fir) != 1.0) {
			debug.println(0, "testDoubleGaussChi2: similarity to self failed!");
			return false;
		}
		
		if(kernel.valueOf(fir, sec) >= 1.) {
			debug.println(0, "testDoubleGaussChi2: orthogonal samples similarity failed!");
			return false;
		}
		
		return true;
	}
	

}
