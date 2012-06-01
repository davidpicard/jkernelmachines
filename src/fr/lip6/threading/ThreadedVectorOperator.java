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
package fr.lip6.threading;

import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Utility for the parallelization of vector operations.
 * @author picard
 *
 */
public abstract class ThreadedVectorOperator {
	
	/**
	 * get the parallelized vector
	 * @param vector double[] instance on which to do the processing
	 * @return resulting vector.
	 */
	public double[] getVector(final double[] vector)
	{
		try
		{
			//one job per line of the matrix
			ThreadPoolExecutor threadPool = ThreadPoolServer.getThreadPoolExecutor();
			Queue<Future<?>> futures = new LinkedList<Future<?>>();
			
			int nbcpu = Runtime.getRuntime().availableProcessors();
			int increm = vector.length / nbcpu + 1;
			
			for(int i = 0 ; i < vector.length ; i += increm)
			{
				final int min = i;
				final int max = Math.min(vector.length, i+increm);
				Runnable r = new Runnable(){
					@Override
					public void run() {
						doBlock(min, max, vector);
					}
				};
				
				futures.add(threadPool.submit(r));
			}

			//wait for all jobs
			while(!futures.isEmpty())
				futures.remove().get();
			
			return vector;
		} catch (InterruptedException e) {

			System.err.println("MatrixWorkerFactory : getMatrix impossible");
			e.printStackTrace();
			return null;
		} catch (ExecutionException e) {
			System.err.println("MatrixWorkerFactory : Exception in execution, matrix unavailable.");
			e.printStackTrace();
			return null;
		}
	}
	
	public abstract void doBlock(int min, int max, final double[] vector);
	
}
