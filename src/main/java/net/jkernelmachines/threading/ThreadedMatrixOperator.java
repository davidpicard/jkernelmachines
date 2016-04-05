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
