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
package fr.lip6.jkernelmachines.type;

import java.io.Serializable;

/**
 * Simple class of training sample that contains the generic &lt; T &gt;  of sample and the associated label.
 * @author dpicard
 *
 * @param <T> data type of input space
 */
public class TrainingSample<T> implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 874733576041102410L;
	
	
	public T sample;
	public int label;
	
	public TrainingSample(T t, int l)
	{
		sample = t;
		label = l;
	}
}
