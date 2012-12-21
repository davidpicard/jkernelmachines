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
package fr.lip6.jkernelmachines.kernel.typed;

import fr.lip6.jkernelmachines.kernel.Kernel;

/**
 * Linear Kernel on float[].
 * @author dpicard
 *
 */
public class FloatLinear extends Kernel<float[]> {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -8274638352733867140L;

	@Override
	public double valueOf(float[] t1, float[] t2) {
		float sum = 0.f;
		int min = Math.min(t1.length, t2.length);
		int i = 0;
		
		while(i < min) {
			sum += t2[i]*t1[i];
			i += 1;
		}
		
		return sum;
	}

	@Override
	public double valueOf(float[] t1) {
		return valueOf(t1, t1);
	}


}
