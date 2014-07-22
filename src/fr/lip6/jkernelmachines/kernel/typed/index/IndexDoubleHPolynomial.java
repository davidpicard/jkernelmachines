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
package fr.lip6.jkernelmachines.kernel.typed.index;

import static java.lang.Math.pow;
import fr.lip6.jkernelmachines.kernel.Kernel;

/**
 * Kernel on double[] that performs the product of a specified component j to the power d:<br />
 * k(x,y) = (0.5 + 0.5*x[j]*y[j])^d
 * @author dpicard
 *
 */
public class IndexDoubleHPolynomial extends Kernel<double[]> {
	
	private static final long serialVersionUID = 3920964082325378818L;
	private int ind = 0;
	private int degree = 1;
	
	/**
	 * Constructor specifying the component which is used
	 * @param feature the index of the component
	 * @param degree the degree of the polynomial
	 */
	public IndexDoubleHPolynomial(int feature, int degree)
	{
		ind = feature;
		this.degree = degree;
	}
	
	@Override
	public double valueOf(double[] t1, double[] t2) {
		if(t1[ind] == 0. || t2[ind] == 0.)
			return 0.;
		return pow(0.5+0.5*t2[ind]*t1[ind], degree);
	}

	@Override
	public double valueOf(double[] t1) {

		if(t1[ind] == 0.)
			return 0.;
		return pow(0.5+0.5*t1[ind]*t1[ind], degree);
	}

	
	public void setIndex(int i)
	{
		this.ind = i;
	}

}
