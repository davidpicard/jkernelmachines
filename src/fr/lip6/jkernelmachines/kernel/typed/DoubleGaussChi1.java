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

import fr.lip6.jkernelmachines.kernel.GaussianKernel;

/**
 * Gaussian Kernel on double[] that uses a Chi1 distance.
 * @author dpicard
 *
 */
public class DoubleGaussChi1 extends GaussianKernel<double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -1011192915083821659L;
	
	
	private double gamma = 0.1;
	private double eps = 10e-8;
	
	@Override
	public double valueOf(double[] t1, double[] t2) {
		double sum = 0.;
		double tmp = 0.;
		
		int lim = Math.min(t1.length, t2.length);
		for (int i = lim-1; i >= 0 ; i--)
			//assume X and Y > 0
			if( (tmp = t1[i]+t2[i]) > eps)
				sum += Math.abs(t1[i] - t2[i]) / tmp; //chi1
		
		return Math.exp(-gamma * sum);
	}

	@Override
	public double valueOf(double[] t1) {
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
	public double distanceValueOf(double[] t1, double[] t2) {
		double sum = 0.;
		double tmp = 0.;
		
		int lim = Math.min(t1.length, t2.length);
		for (int i = lim-1; i >= 0 ; i--)
			//assume X and Y > 0
			if( (tmp = t1[i]+t2[i]) > eps)
				sum += Math.abs(t1[i] - t2[i]) / tmp; //chi1
		return sum;
	}

}
