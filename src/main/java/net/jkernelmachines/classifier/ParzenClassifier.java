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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import net.jkernelmachines.kernel.Kernel;
import net.jkernelmachines.type.TrainingSample;

/**
 * Classification tool using a Parzen window
 * @author dpicard
 *
 * @param <T> type of input space
 */
public class ParzenClassifier<T> implements Classifier<T>, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5147554432765939157L;
	
	
	Kernel<T> kernel;
	ArrayList<TrainingSample<T>> ts;
	
	public ParzenClassifier(Kernel<T> kernel)
	{
		this.kernel = kernel;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#train(java.lang.Object, int)
	 */
	public void train(TrainingSample<T> t) {

		if(ts == null)
		{
			ts = new ArrayList<TrainingSample<T>>();
			
		}
		ts.add(t);

		
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#train(T[], int[])
	 */
	public void train(List<TrainingSample<T>> t) {

		ts = new ArrayList<TrainingSample<T>>(t);
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#valueOf(java.lang.Object)
	 */
	public double valueOf(T e) {

		double sum = 0.;
		for(int i = 0 ; i < ts.size(); i++)
		{
			TrainingSample<T> t = ts.get(i);
			sum += t.label * kernel.valueOf(t.sample, e);
		}
		
		sum /= ts.size();
		
		return sum;
	}
	

	/**
	 * Creates and returns a copy of this object.
	 * @see java.lang.Object#clone()
	 */
	@SuppressWarnings("unchecked")
	@Override
	public ParzenClassifier<T> copy() throws CloneNotSupportedException {
		return (ParzenClassifier<T>) super.clone();
	}
	
}
