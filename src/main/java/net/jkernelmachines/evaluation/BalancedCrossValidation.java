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
package net.jkernelmachines.evaluation;

/**
 * Interface for crossvalidation classes that can balance the number of sample
 * per class while computing the splits.
 * 
 * @author picard
 * 
 */
public interface BalancedCrossValidation {
	

	/**
	 * Returns true if the splits are balanced between positive and negative
	 * @return
	 */
	public boolean isBalanced();


	/**
	 * Set class balancing strategy when computing the splits
	 * @param balanced true if enables balancing
	 */
	public void setBalanced(boolean balanced);


}
