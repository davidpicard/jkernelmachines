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
package net.jkernelmachines.kernel.typed.index;

import net.jkernelmachines.kernel.GaussianKernel;

/**
 * Kernel on double[] that computes the Chi2 distance of a specified component j:
 * k(x, y) = (x[j]-y[j])*(x[j]-y[j])/(x[j]+y[j])
 * @author dpicard
 *
 */
public class IndexDoubleGaussChi2 extends GaussianKernel<double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 102467593724674738L;
	
	
	private double gamma = 1.0;
	private int ind = 0;
	private double eps = 10e-6;
	
	/**
	 * Constructor specifying the component which is used
	 * @param feature the index of the component
	 */
	public IndexDoubleGaussChi2(int feature)
	{
		ind = feature;
	}
	
	@Override
	public double valueOf(double[] t1, double[] t2) {
		if(t1[ind] == 0. && t2[ind] == 0.)
			return 1.;
		
		double tmp = 0;
		double min = 0;
		
		//assume X and Y > 0
		if( (tmp = t1[ind]+t2[ind]) > eps)
		{
			min = t1[ind]-t2[ind];
			min = (min*min) / tmp; //chi2
		}
	
		return Math.exp(-gamma * min);
	}

	@Override
	public double valueOf(double[] t1) {
		
		return 1.0;
	}

	public void setGamma(double g)
	{
		gamma = g;
	}
	
	public void setIndex(int i)
	{
		this.ind = i;
	}

	public double getGamma() {
		return gamma;
	}

	@Override
	public double distanceValueOf(double[] t1, double[] t2) {
		if(t1[ind] == 0. && t2[ind] == 0.)
			return 1.;
		
		double tmp = 0;
		double min = 0;
		
		//assume X and Y > 0
		if( (tmp = t1[ind]+t2[ind]) > eps)
		{
			min = t1[ind]-t2[ind];
			min = (min*min) / tmp; //chi2
		}
		return min;
	}
}
