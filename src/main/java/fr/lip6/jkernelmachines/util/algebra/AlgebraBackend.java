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
package fr.lip6.jkernelmachines.util.algebra;

/**
 * <p>
 * Class that allows external computation of eig and inv. The use is to
 * instanciate a new subclass of this class related to a specific matrix labrary
 * and to register it.
 * </p>
 * 
 * @author picard
 *
 */
public abstract class AlgebraBackend {

	/**
	 * Compute the inverse matrix
	 * 
	 * @param A
	 *            input matrix
	 * @return the inverse of A if possible
	 */
	public abstract double[][] inv(final double[][] A);
	/**
	 * Performs the eigen decomposition of a symmetric matrix:
	 * A = Q * L * Q'
	 * with Q orthonormal and L diagonal
	 * @param A input matrix
	 * @return an array of two matrices containing {Q, L}
	 */
	public abstract double[][][] eig(final double[][] A);

}
