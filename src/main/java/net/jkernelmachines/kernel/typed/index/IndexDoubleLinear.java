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

import net.jkernelmachines.kernel.Kernel;

/**
 * Kernel on double[] that performs the product of a specified component j:
 * k(x,y) = x[j]*y[j]
 * @author dpicard
 *
 */
public class IndexDoubleLinear extends Kernel<double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 457509780005148420L;
	
	
	private int ind = 0;
	
	/**
	 * Constructor specifying the component which is used
	 * @param feature the index of the component
	 */
	public IndexDoubleLinear(int feature)
	{
		ind = feature;
	}
	
	@Override
	public double valueOf(double[] t1, double[] t2) {
		if(t1[ind] == 0. || t2[ind] == 0.)
			return 0.;
		return t2[ind]*t1[ind];
	}

	@Override
	public double valueOf(double[] t1) {

		if(t1[ind] == 0.)
			return 0.;
		return t1[ind]*t1[ind];
	}

	
	public void setIndex(int i)
	{
		this.ind = i;
	}

}
