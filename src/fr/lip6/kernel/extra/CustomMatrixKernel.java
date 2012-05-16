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
package fr.lip6.kernel.extra;

import fr.lip6.kernel.Kernel;

/**
 * <p>
 * Kernel with a provided custom matrix.
 * </p>
 * <p>
 * The datatype of input space is integer relative to row/column indices. Therefore, the similarity
 * between elements i and j is matrix[i][j]. <br />
 * If i or j is not in the range of the matrix, 0 is returned.
 * </p>
 * @author dpicard
 *
 */
public class CustomMatrixKernel extends Kernel<Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -5379932592270965091L;
	private double matrix[][];
	
	/**
	 * Constructor using the supplied Gram matrix.
	 * @param matrix the Gram matrix of underlying kernel function.
	 */
	public CustomMatrixKernel(double matrix[][])
	{
		this.matrix = matrix;
	}
	
	@Override
	public double valueOf(Integer t1, Integer t2) {
		if(t1 > matrix.length || t2 > matrix.length)
			return 0;
		return matrix[t1][t2];
	}

	@Override
	public double valueOf(Integer t1) {
		if(t1 > matrix.length)
			return 0.;
		return matrix[t1][t1];
	}

}
