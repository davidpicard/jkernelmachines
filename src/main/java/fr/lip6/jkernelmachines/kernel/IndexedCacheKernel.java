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

    Copyright David Picard - 2010

*/
package fr.lip6.jkernelmachines.kernel;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import fr.lip6.jkernelmachines.threading.ThreadedMatrixOperator;

/**
 * <p>
 * Simple method of pair &lt; Key,Value &gt; for caching a kernel function.
 * </p>
 * <p>
 * A unique Key is assigned to each sample, and similarities between two samples are cached in Hashmaps using the keys.
 * Warning: Safety granted by assertion. Disable assertion for a faster but riskier execution.
 * </p>
 * @author picard
 *
 * @param <S> The datatype of key (usually String or Integer)
 * @param <T> The data type of input space
 */
public final class IndexedCacheKernel<S,T> extends Kernel<S> {

	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1706425748759430692L;
	
	private double[][] matrix;
	private HashMap<S, Integer> map;
	final private transient Kernel<T> kernel;
	
	/**
	 * Constructor using an underlying kernel and a map of &lt; Key, Sample &gt;
	 * @param k the underlying kernel
	 * @param signatures the map giving the key associated with each sample
	 */
	public IndexedCacheKernel(Kernel<T> k, final Map<S, T> signatures)
	{
		this.kernel = k;
		
		matrix = new double[signatures.size()][signatures.size()];
	
		//adding index
		map = new HashMap<S, Integer>(signatures.size());
		final HashMap<Integer, S> rmap = new HashMap<Integer, S>(signatures.size());
		int index = 0;
		for(S s : signatures.keySet())
		{
			map.put(s, index);
			rmap.put(index, s);
			index++;
		}
		
		//computing matrix				
		ThreadedMatrixOperator factory = new ThreadedMatrixOperator()
		{
			@Override
			public void doLines(double[][] matrix, int from, int to) {
				for(int index = from ; index < to ; index++)
				{
					//reverse search through mapping S <-> index
					S s1 = rmap.get(index);
					//mapped signature
					T t1 = signatures.get(s1);

					//all mapping S <-> T
					for(Iterator<S> iter = map.keySet().iterator() ; iter.hasNext() ;)
					{
						S s2 = iter.next();
						//get index of s2
						int j = map.get(s2);
						//get signature of s2
						T t2 = signatures.get(s2);
						//add value of kernel
						matrix[index][j] = kernel.valueOf(t1, t2);
					}
				}
			};
		};


		/* do the actuel computing of the matrix */
		matrix = factory.getMatrix(matrix);
				
	}
	
	@Override
	public final double valueOf(S t1, S t2) {
		//return 0 if doesn't know of
		assert(map.containsKey(t1) && map.containsKey(t2));
		
//		{
//			System.err.println("<"+t1+","+t2+"> not in matrix !!!");
//			return 0;
//		}
		int id1 = map.get(t1);
		int id2 = map.get(t2);
		
		return ((double)(float)matrix[id1][id2]);
	}

	@Override
	public double valueOf(S t1) {
		//return 0 if doesn't know of
		if(!map.containsKey(t1))
		{
			System.err.println("<"+t1+","+t1+"> not in matrix !!!");
			return 0;
		}
		
		int id = map.get(t1);
		return ((double)(float)matrix[id][id]);
	}
	
	public Map<S, Integer> getMap() {
		return map;
	}
	
	public double[][] getCacheMatrix() {
		return matrix;
	}

}
