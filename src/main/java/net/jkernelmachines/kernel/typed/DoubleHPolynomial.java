/*******************************************************************************
 * Copyright (c) 2016, David Picard.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *******************************************************************************/

package net.jkernelmachines.kernel.typed;

import net.jkernelmachines.kernel.Kernel;
import net.jkernelmachines.util.algebra.VectorOperations;
import static java.lang.Math.pow;

/**
 * Homogenous Polynomial kernel on double arrays. return (x'y)^d, where d = 2 by default
 * @author David Picard
 */
public class DoubleHPolynomial extends Kernel<double[]> {

	private static final long serialVersionUID = 2961358288571394251L;
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
