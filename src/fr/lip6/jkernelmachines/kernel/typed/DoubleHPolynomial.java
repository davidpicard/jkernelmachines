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

package fr.lip6.jkernelmachines.kernel.typed;

import fr.lip6.jkernelmachines.kernel.Kernel;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;
import static java.lang.Math.pow;

/**
 * Homogenous Polynomial kernel on double arrays. return (x'y)^d, where d = 2 by default
 * @author David Picard
 */
public class DoubleHPolynomial extends Kernel<double[]> {

    private int d = 2;
    
    /**
     * Default constructor, d = 2
     */
    public DoubleHPolynomial() {
        d = 2;
    }
    
    /**
     * Constructor specifying the degree of the polynomial kernel
     * @param degree the exponent to which the dot product is raised
     */
    public DoubleHPolynomial(int degree) {
        d = degree;
    }
    
    @Override
    public double valueOf(double[] t1, double[] t2) {
        return pow(VectorOperations.dot(t1, t2), d);
    }

    @Override
    public double valueOf(double[] t1) {
        return pow(VectorOperations.dot(t1, t1), d);
    }
    
}
