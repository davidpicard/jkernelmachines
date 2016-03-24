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

import static java.lang.Math.sqrt;

import java.util.HashMap;

import fr.lip6.jkernelmachines.kernel.Kernel;

/**
 * @author picard
 *
 */
public class NormalizedStringNGram extends Kernel<String> {

	/**
	 * 
	 */
	private static final long serialVersionUID = -6437260079523365632L;
	int n = 2;

	public NormalizedStringNGram(int n) {
		this.n = n;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.kernel.Kernel#valueOf(java.lang.Object,
	 * java.lang.Object)
	 */
	@Override
	public double valueOf(String t1, String t2) {
		double r = 0;

		HashMap<String, Double> m1 = new HashMap<>();
		for (int i = 0; i < t1.length() - n + 1; i++) {
			String s = t1.substring(i, i + n);
			double d = m1.containsKey(s) ? m1.get(s) : 0;
			m1.put(s, d + 1);
		}
		double r1 = 0;
		for (double d : m1.values()) {
			r1 += d * d;
		}
		if( r1 == 0) {
			return 0;
		}
		
		HashMap<String, Double> m2 = new HashMap<>();
		for (int i = 0; i < t2.length() - n + 1; i++) {
			String s = t2.substring(i, i + n);
			double d = m2.containsKey(s) ? m2.get(s) : 0;
			m2.put(s, d + 1);
		}
		double r2 = 0;
		for (double d : m2.values()) {
			r2 += d * d;
		}
		if(r2 == 0) {
			return 0;
		}

		if (m2.size() < m1.size()) {
			for (String s : m2.keySet()) {
				if (m1.containsKey(s)) {
					r += m1.get(s) * m2.get(s);
				}
			}
		} else {
			for (String s : m1.keySet()) {
				if (m2.containsKey(s)) {
					r += m2.get(s) * m1.get(s);
				}
			}
		}

		return r / sqrt(r1 * r2);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.kernel.Kernel#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(String t1) {
		return 1;
	}

}
