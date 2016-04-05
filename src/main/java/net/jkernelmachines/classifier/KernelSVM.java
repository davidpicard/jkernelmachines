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
package net.jkernelmachines.classifier;

import net.jkernelmachines.kernel.Kernel;

/**
 * Interface for SVM algorithms using a non-linear kernel (dual optimization mainly)
 * 
 * @author picard
 *
 */
public interface KernelSVM<T> extends Classifier<T> {
    
        /**
         * Tells the current Kernel.
         * @return the current kernel
         */
        public Kernel<T> getKernel();
	
	/**
	 * Sets the kernel to use as similarity measure
	 * @param k the kernel function
	 */
	public void setKernel(Kernel<T> k);
	
	/**
	 * Tells the weights of training samples
	 * 
	 * @return an array of double representing the weights
	 */
	public double[] getAlphas();
	
	/**
	 * Sets the hyperparameter C
	 * @param c the hyperparameter C
	 */
	public void setC(double c);
	
	/**
	 * Tells the hyperparameter C
	 * @return the hyperparameter C
	 */
	public double getC();

}
