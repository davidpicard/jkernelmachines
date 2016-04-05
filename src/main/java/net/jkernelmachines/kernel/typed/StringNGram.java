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

import java.util.HashMap;

import net.jkernelmachines.kernel.Kernel;

/**
 * @author picard
 *
 */
public class StringNGram extends Kernel<String> {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -6437260079523365632L;
	int n = 2;
	
	public StringNGram(int n) {
		this.n = n;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.kernel.Kernel#valueOf(java.lang.Object, java.lang.Object)
	 */
	@Override
	public double valueOf(String t1, String t2) {
		double r = 0;
		String s1, s2;
		
		if(t1.length() < t2.length()) {
			s1 = t1;
			s2 = t2;
		}
		else {
			s1 = t2; 
			s2 = t1;
		}
		
		HashMap<String, Double> m1 = new HashMap<>();
		for(int i = 0 ; i < s1.length()-n+1 ; i++) {
			String s = s1.substring(i, i+n);
			double d = m1.containsKey(s)?m1.get(s):0;
			m1.put(s, d+1);
		}

		HashMap<String, Double> m2 = new HashMap<>();
		for(int i = 0 ; i < s2.length()-n+1 ; i++) {
			String s = s2.substring(i, i+n);
			if(m1.containsKey(s)) {
				double d = m2.containsKey(s)?m2.get(s):0;
				m2.put(s, d+1);
			}
		}
		
		for(String s : m2.keySet()) {
			double d = m2.get(s);
			r += m1.get(s)*d;
		}
		
		return r;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.kernel.Kernel#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(String t1) {
		return valueOf(t1, t1);
	}

}
