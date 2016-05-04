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

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
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
	
	/**
	 * Generate the set of all possible n-grams composed of characters taken from the input string
	 * @param alp the original alphabet
	 * @param n dimension of the tuples
	 * @return map of possible n-grams and corresponding indexes
	 */
	public static Map<String, Integer> generateNGramAlphabet(String alp, int n) {
		Map<String, Integer> set = new HashMap<>();
		
		if(n <= 1) {
			for(int i = 0 ; i < alp.length() ; i++) {
				String s = ""+alp.charAt(i);
				set.put(s, i);
			}
			return set;
		}
		else {
			int index = 0;
			Map<String, Integer> prev = generateNGramAlphabet(alp, n-1);
			for(int i = 0 ; i < alp.length() ; i++) {
				String a = ""+alp.charAt(i);
				for(String s : prev.keySet()) {
					set.put(a+s, index++);
				}
			}
			return set;
		}
	}
	
	/**
	 * computes the histogram of occurrences of n-grams gien in the alphabet
	 * @param s string to compute the histoogram
	 * @param m alphabet of n-grams and corresponding indexes in the output vector
	 * @return histogram
	 */
	public static double[] projectNGramAlphabet(String s, Map<String, Integer> m) {
		double[] res = new double[m.size()];
		
		// assume all n-gram have same size
		int n = m.keySet().iterator().next().length();
		
		for(int i = 0 ; i < s.length()-n ; i++) {
			String k = s.substring(i, i+n);
			if(m.containsKey(k)) {
				res[m.get(k)]++;
			}
		}
		return res;
	}
	
	/**
	 * Generates the alphabet of n-grams that have a minimal number of occurences in a list of strings
	 * @param l the list of strings
	 * @param n n-gram
	 * @param thresh minimum number of occurences of the n-grams in the list
	 * @return
	 */
	public static Map<String, Integer> generateMinimumNGramAlphabet(List<String> l, int n, int thresh) {
		
		Map<String, Double> occ = computeNGram(l.get(0), n);
		for(int i = 1 ; i < l.size() ; i++) {
			Map<String, Double> m = computeNGram(l.get(i), n);
			for(String s : m.keySet()) {
				if(occ.containsKey(s)) {
					occ.put(s, occ.get(s)+m.get(s));
				}
				else {
					occ.put(s,  m.get(s));
				}
			}
		}
		
		int index = 0;
		Map<String, Integer> map = new HashMap<>();
		for(String k : occ.keySet()) {
			if(occ.get(k) >= thresh) {
				map.put(k, index++);
			}
		}
		
		return map;
	}
	
	/**
	 * Generates the list of the most frequent n-grams in a list of string
	 * @param l the list
	 * @param n n-gram
	 * @param nb number of most frequent n-grams
	 * @return
	 */
	public static Map<String, Integer> generateMostFrequentNGramAlphabet(List<String> l , int n, int nb) {
		Map<String, Double> occ = computeNGram(l.get(0), n);
		for(int i = 1 ; i < l.size() ; i++) {
			Map<String, Double> m = computeNGram(l.get(i), n);
			for(String s : m.keySet()) {
				if(occ.containsKey(s)) {
					occ.put(s, occ.get(s)+m.get(s));
				}
				else {
					occ.put(s,  m.get(s));
				}
			}
		}
		
		int thresh = 0;
		if(occ.size() > nb) {
			List<Double> sorted = new ArrayList<>(occ.values());
			Collections.sort(sorted);
		
			thresh = (int)sorted.get(sorted.size() - nb).doubleValue();
		}
	
		
		int index = 0;
		Map<String, Integer> map = new HashMap<>();
		for(String k : occ.keySet()) {
			if(occ.get(k) > thresh) {
				map.put(k, index++);
			}
		}
		
		return map;
	}

}
