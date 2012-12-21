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
package fr.lip6.jkernelmachines.kernel.adaptative;

import java.util.ArrayList;

import fr.lip6.jkernelmachines.kernel.Kernel;

/**
 * performs a weighted product of several minor kernels, non threaded version.
 * @see ThreadedProductKernel
 * 
 * @author dpicard
 *
 * @param <T> data type of input space.
 */
public class WeightedProductKernel<T> extends Kernel<T> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 6273022923321895693L;
	
	ArrayList<Kernel<T>> kernels;
	ArrayList<Double> weights;
	
	public WeightedProductKernel ()
	{
		super();
		
		kernels = new ArrayList<Kernel<T>>();
		weights = new ArrayList<Double>();
	}
	
	@Override
	public double valueOf(T t1, T t2) {
		
		double prod = 1.0;
		
		for(int i = 0 ; i < kernels.size(); i++)
			prod *= Math.pow(kernels.get(i).valueOf(t1, t2), weights.get(i));
		
		return prod;
	}

	@Override
	public double valueOf(T t1) {
		return valueOf(t1, t1);
	}
	
	/**
	 * adds a kernel to to product
	 * @param k
	 */
	public void addKernel(Kernel<T> k)
	{
		kernels.add(k);
		weights.add(1.0);
	}
	
	public void addKernel(Kernel<T> k, double w)
	{
		kernels.add(k);
		weights.add(w);
	}
	
	/**
	 * removes a kernel from the product
	 * @param k
	 */
	public void removeKernel(Kernel<T> k)
	{
		kernels.remove(k);
	}

}
