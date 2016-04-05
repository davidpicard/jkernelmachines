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
package net.jkernelmachines.type;

import java.util.List;

/**
 * Stream based on a list of samples
 * @author picard
 *
 */
public class ListSampleStream<T> implements TrainingSampleStream<T> {
	
	List<TrainingSample<T>> list;
	int index;
	int e;
	int E = 1;
	
	/**
	 * Constructor specifying the list from which the stream is created 
	 */
	public ListSampleStream(List<TrainingSample<T>> l) {
		this.list = l;
		index = 0;
		e = 0;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.type.TrainingSampleStream#nextSample()
	 */
	@Override
	public TrainingSample<T> nextSample() {
		if(e < E) {
			if(index < list.size()) {
				return list.get(index++);
			}
			else
			{
				index = 0;
				e++;
				return nextSample();
			}
		}
		return null;
	}

	/**
	 * Get the number of times the list is passed through
	 * @return the number of epochs
	 */
	public int getE() {
		return E;
	}

	/**
	 * Sets the number of times the list is passed through (number of epochs)
	 * @param e
	 */
	public void setE(int e) {
		E = e;
	}

}
