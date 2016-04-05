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
