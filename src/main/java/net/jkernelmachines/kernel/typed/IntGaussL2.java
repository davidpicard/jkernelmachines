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
package net.jkernelmachines.kernel.typed;

import net.jkernelmachines.kernel.GaussianKernel;

/**
 * Gaussian Kernel on int[] that uses a L2 distance.
 * @author dpicard
 *
 */
public class IntGaussL2 extends GaussianKernel<int[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4415295327411182596L;
	
	
	private double gamma = 0.1;
	
	@Override
	public double valueOf(int[] t1, int[] t2) {
		double sum = 0.;
		for(int i = 0 ; i < Math.min(t1.length, t2.length); i++)
			sum += (t1[i]-t2[i])*(t1[i] - t2[i]);
		return Math.exp(-gamma * sum);
	}

	@Override
	public double valueOf(int[] t1) {
		return 1.0;
	}


	/**
	 * @return the sigma
	 */
	public double getGamma() {
		return gamma;
	}

	/**
	 * @param gamma inverse of std dev parameter
	 */
	public void setGamma(double gamma) {
		this.gamma = gamma;
	}

	@Override
	public double distanceValueOf(int[] t1, int[] t2) {
		double sum = 0.;
		for(int i = 0 ; i < Math.min(t1.length, t2.length); i++)
			sum += (t1[i]-t2[i])*(t1[i] - t2[i]);
		return sum;
	}
}
