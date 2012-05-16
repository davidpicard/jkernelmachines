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
package fr.lip6.density;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import fr.lip6.kernel.Kernel;

/**
 * Parzen window for estimating the probability density function of a random variable.
 * @author dpicard
 *
 * @param <T> Datatype of input space
 */
public class ParzenDensity<T> implements DensityFunction<T>, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5414922333951533146L;
	
	
	private Kernel<T> kernel;
	ArrayList<T> set;
	
	public ParzenDensity(Kernel<T> kernel)
	{
		this.kernel = kernel;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.density.DensityFunction#train(java.lang.Object)
	 */
	public void train(T e) {
		if(set == null)
		{
			set = new ArrayList<T>();
		}

		set.add(e);
				
	}

	/* (non-Javadoc)
	 * @see fr.lip6.density.DensityFunction#train(T[])
	 */
	public void train(List<T> e) {
		if(set == null)
		{
			set = new ArrayList<T>();
		}
		
		for(T t : e)
			set.add(t);
	}

	/* (non-Javadoc)
	 * @see fr.lip6.density.DensityFunction#valueOf(java.lang.Object)
	 */
	public double valueOf(T e) {

		double sum = 0.;
		for(int i = 0 ; i < set.size(); i++)
			sum += kernel.valueOf(set.get(i), e);
		
		sum /= set.size();
		
		return sum;
	}

}
