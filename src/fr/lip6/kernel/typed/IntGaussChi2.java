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
package fr.lip6.kernel.typed;

import fr.lip6.kernel.GaussianKernel;

/**
 * Gaussian Kernel on int[] that uses a Chi2 distance.
 * @author dpicard
 *
 */
public class IntGaussChi2 extends GaussianKernel<int[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7324980870969389374L;
	
	
	private double gamma = 0.1;
	private double eps = 0;
	
	@Override
	public double valueOf(int[] t1, int[] t2) {
		double sum = 0.;
		double tmp = 0.;
		for (int i = 0; i < Math.min(t1.length, t2.length); i++)
			//assume X and Y > 0
			if( (tmp = t1[i]+t2[i]) > eps)
				sum += (t1[i] - t2[i])*(t1[i] - t2[i]) / tmp; //chi2
		
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
		double tmp = 0.;
		for (int i = 0; i < Math.min(t1.length, t2.length); i++)
			//assume X and Y > 0
			if( (tmp = t1[i]+t2[i]) > eps)
				sum += (t1[i] - t2[i])*(t1[i] - t2[i]) / tmp; //chi2
		return sum;
	}
}
