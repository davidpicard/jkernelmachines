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
/**
 * 
 */
package net.jkernelmachines.projection;

import java.util.HashMap;
import java.util.Map;

/**
 * Utility class that computes histograms of n-grams from a string
 * 
 * @author picard
 *
 */
public class StringNGramProjection {

	/**
	 * Computes the histogram of n-grams
	 * 
	 * @param input
	 *            input string
	 * @param n
	 *            length of considered sub-sequences
	 * @return map with n-gram as keys and number of occurences as values
	 */
	public static Map<String, Double> computeNGram(String input, int n) {
		HashMap<String, Double> m1 = new HashMap<>();
		for (int i = 0; i < input.length() - n + 1; i++) {
			String s = input.substring(i, i + n);
			double d = m1.containsKey(s) ? m1.get(s) : 0;
			m1.put(s, d + 1);
		}
		return m1;
	}

	/**
	 * Computes the histogram of n-grams summing to 1
	 * 
	 * @param input
	 *            input string
	 * @param n
	 *            length of considered sub-sequences
	 * @return map with n-gram as keys and normalized number of occurences as
	 *         values
	 */
	public static Map<String, Double> computeNormalizedNGram(String input, int n) {
		HashMap<String, Double> m1 = new HashMap<>();
		double total = 0;
		for (int i = 0; i < input.length() - n + 1; i++) {
			String s = input.substring(i, i + n);
			double d = m1.containsKey(s) ? m1.get(s) : 0;
			m1.put(s, d + 1);
			total++;
		}

		for (String s : m1.keySet()) {
			m1.put(s, m1.get(s) / total);
		}

		return m1;
	}

}
