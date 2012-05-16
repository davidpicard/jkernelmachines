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
 * Gaussian Kernel on double[] that uses a L2 distance.
 * @author dpicard
 *
 */
public class DoubleTriangleL2 extends GaussianKernel<double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8139656729530005699L;
	
	
	private double gamma = 0.1;
	
	public DoubleTriangleL2(double g) {
		gamma = g;
	}

	public DoubleTriangleL2() {
	}

	@Override
	public double valueOf(double[] t1, double[] t2) {
		double sum = 0.;
		int lim = Math.min(t1.length, t2.length);
		for(int i = lim-1 ; i >= 0 ; i--)
		{
			double d = (t1[i]-t2[i]);
			sum += d*d;
		}
		if(Double.isNaN(sum))
		{
			System.err.println(this+" : Warning sum NaN");
			return 0.0;
		}
		return Math.max(0, 1 -gamma * sum);
	}

	@Override
	public double valueOf(double[] t1) {
		return valueOf(t1, t1);
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
		int lim = Math.min(t1.length, t2.length);
		for(int i = lim-1 ; i >= 0 ; i--)
		{
			double d = (t1[i]-t2[i]);
			sum += d*d;
		}
		if(Double.isNaN(sum))
		{
			System.err.println(this+" : Warning sum NaN");
			return 0.0;
		}
		return sum;
	}
}
