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
package net.jkernelmachines.threading;

import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

import net.jkernelmachines.util.DebugPrinter;

/**
 * Utility for the parallelization of matrix operations.
 * @author picard
 *
 */
public abstract class ThreadedMatrixOperator {
	
	static DebugPrinter debug = new DebugPrinter();
	
	static int lines = 1;
	
	/**
	 * get the parallelized matrix
	 * @param matrix double[][] on which the processing is done
	 * @return the resulting matrix
	 */
	public double[][] getMatrix(final double[][] matrix)
	{
		//one job per line of the matrix
		ThreadPoolExecutor threadPool = ThreadPoolServer.getThreadPoolExecutor();
		Queue<Future<?>> futures = new LinkedList<Future<?>>();
		
		int increm = lines;
		
		try
		{
			for(int i = 0 ; i < matrix.length ; i += increm)
			{
				final int from = i;
				final int to = Math.min(matrix.length, i+increm);
				
				
				Runnable r = new Runnable(){
					public void run() {
						doLines(matrix, from, to);
					}
				};
				
				futures.add(threadPool.submit(r));
			}

			//wait for all jobs
			while(!futures.isEmpty())
			{
				futures.remove().get();
			}

			ThreadPoolServer.shutdownNow(threadPool);
			
			return matrix;
		} catch (InterruptedException e) {

			debug.println(3, "MatrixWorkerFactory : getMatrix interrupted");
			return null;
		} catch (ExecutionException e) {
			debug.println(1, "MatrixWorkerFactory : Exception in execution, matrix unavailable.");
			e.printStackTrace();
			return null;
		}
		
	}
	
	public abstract void doLines(double matrix[][], int from, int to);
	
	/**
	 * Sets the number of lines computed by each job
	 * @param n the number of lines (default 1)
	 */
	public static void setLines(int n) {
		lines = n;
	}
}
