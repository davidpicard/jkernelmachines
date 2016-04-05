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

/**
 * Interface to streams of training samples, useful for online training.
 * 
 * @author picard
 *
 */
public interface TrainingSampleStream<T> {
	
	/**
	 * Return the next training sample from this stream
	 * @return null if the stream has no more elements
	 */
	public TrainingSample<T> nextSample();

}
