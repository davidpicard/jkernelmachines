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
package fr.lip6.jkernelmachines.threading;

import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Utility for the parallelization of matrix operations.
 * @author picard
 *
 */
public abstract class ThreadedMatrixOperator {
	

	//one job per line of the matrix
	ThreadPoolExecutor threadPool = ThreadPoolServer.getThreadPoolExecutor();
	Queue<Future<?>> futures = new LinkedList<Future<?>>();
	
	int nbjobs = (2*Runtime.getRuntime().availableProcessors());
	
	/**
	 * get the parallelized matrix
	 * @param matrix double[][] on which the processing is done
	 * @return the resulting matrix
	 */
	public double[][] getMatrix(final double[][] matrix)
	{
		int increm = matrix.length / nbjobs  + 1 ;
		
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
//				System.out.print("\r"+futures.size()+" to go");
			}
//			System.out.print("\r");

			return matrix;
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
	
	public abstract void doLines(double matrix[][], int from, int to);
	
	public void setNbJobs(int n) {
		nbjobs = n;
	}
	
}
