/**
 * 
 */
package net.jkernelmachines.kernel.extra.bag;

import static java.lang.Math.min;
import java.util.Map;

import net.jkernelmachines.kernel.Kernel;

/**
 * Kernel that takes sparse histograms as input and computes the dintersection
 * (min). The histograms are represented by a map with the bins as keys (generic
 * type) and a double as value for the corresponding number of elements in that
 * bin.
 * 
 * @author picard
 * @param <T>
 *            input space data type
 *
 */
public class MapIntersectionKernel<T> extends Kernel<Map<T, Double>> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3007151215305286619L;

	/*
	 * (non-Javadoc)
	 * 
	 * @see net.jkernelmachines.kernel.Kernel#valueOf(java.lang.Object,
	 * java.lang.Object)
	 */
	@Override
	public double valueOf(Map<T, Double> t1, Map<T, Double> t2) {
		double r = 0;
		for (T x : t1.keySet()) {
			if (t2.containsKey(x)) {
				r += min(t1.get(x), t2.get(x));
			}
		}
		return r;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see net.jkernelmachines.kernel.Kernel#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(Map<T, Double> t1) {
		double r = 0;
		for (T x : t1.keySet()) {
			double d = t1.get(x);
			r += d;
		}
		return r;
	}

}
