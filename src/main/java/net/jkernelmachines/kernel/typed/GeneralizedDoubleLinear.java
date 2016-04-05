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

import net.jkernelmachines.kernel.Kernel;

/**
 * Generalized  linear kernel on double[]. Provided a proper inner product matrix M, this kernel returns :
 * k(x, y) = x'*M*y
 * @author dpicard
 *
 */
public class GeneralizedDoubleLinear extends Kernel<double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6618610116348247480L;
	
	double[][] M;
	int size;
	
	public GeneralizedDoubleLinear(double[][] innerProduct)
	{
		if(innerProduct.length != innerProduct[0].length)
		{
			M = null;
			size = 0;
		}
		else
		{
			M = innerProduct;
			size = M.length;
		}
		
		
	}
	
	@Override
	public double valueOf(double[] t1, double[] t2) {
		
		if(t1.length != size && t2.length != size)
			return 0;
		
		double sum = 0;
		for(int i = 0 ; i < M.length; i++)
		{
			double xtM = 0;
			for(int j = 0 ; j < M[0].length; j++)
			{
				xtM += t1[j]*M[j][i];
			}
			sum += xtM*t2[i];
		}
		
		return sum;
	}

	@Override
	public double valueOf(double[] t1) {
		
		return valueOf(t1, t1);
	}

}
