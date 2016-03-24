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
package fr.lip6.jkernelmachines.kernel.extra;

import fr.lip6.jkernelmachines.kernel.Kernel;

/**
 * Simple Kernel elevating the underlying kernel to power e in the form K(x,y) = k(x,y)^e.
 * @author picard
 *
 * @param <T> data type of underlying input space.
 */
public class PowerKernel<T> extends Kernel<T> {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 4913325882464601407L;
	
	
	private Kernel<T> kernel;
	private double e = 1.0;
	
	/**
	 * Constructor providing the underlying kernel and the power e
	 * @param kernel the underlying kernel
	 * @param e the power parameter
	 */
	public PowerKernel(Kernel<T> kernel, double e) {
		this.kernel = kernel;
		this.e = e;
	}
	@Override
	public double valueOf(T t1, T t2) {
		return Math.pow(kernel.valueOf(t1, t2), e);
	}
	@Override
	public double valueOf(T t1) {
		return Math.pow(kernel.valueOf(t1), e);
	}
	
	


}
