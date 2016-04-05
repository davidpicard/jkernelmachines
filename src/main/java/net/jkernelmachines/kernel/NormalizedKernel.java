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

    Copyright David Picard - 2014

*/
package net.jkernelmachines.kernel;

/**
 * @author picard
 *
 */
public class NormalizedKernel<T> extends Kernel<T> {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 7265699955337327761L;
	Kernel<T> kernel;
	
	public NormalizedKernel(Kernel<T> k) {
		kernel = k;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.kernel.Kernel#valueOf(java.lang.Object, java.lang.Object)
	 */
	@Override
	public double valueOf(T t1, T t2) {
		return kernel.normalizedValueOf(t1, t2);
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.kernel.Kernel#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(T t1) {
		return kernel.normalizedValueOf(t1, t1);
	}

}
