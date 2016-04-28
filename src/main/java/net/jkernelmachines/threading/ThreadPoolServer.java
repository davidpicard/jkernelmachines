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

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import net.jkernelmachines.util.DebugPrinter;

/**
 * Threading utility used by various algorithm for obtaining a pool of threads.
 * 
 * @author picard
 * 
 */
public class ThreadPoolServer {

	// private static ThreadPoolExecutor executor;
	private static DebugPrinter debug = new DebugPrinter();

	static int nbcpu = Runtime.getRuntime().availableProcessors();

	/**
	 * Tells the system wide ThreadPoolServer (Singleton pattern)
	 * 
	 * @return system wide instance of this class.
	 */
	public static ThreadPoolExecutor getThreadPoolExecutor() {
		ThreadPoolExecutor executor;
		executor = new ThreadPoolExecutor(nbcpu, nbcpu+2, 1,
				TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());
		executor.prestartAllCoreThreads();
		executor.allowCoreThreadTimeOut(true);
		return executor;
	}

	/**
	 * Stops the server.
	 * @param executor the executor to stop
	 */
	public static void shutdownNow(ThreadPoolExecutor executor) {
		if (executor != null) {
			executor.shutdown();
			try {
				while (!executor.isShutdown()) {
					executor.awaitTermination(100, TimeUnit.MILLISECONDS);
				}
			} catch (InterruptedException e) {
				debug.println(1, "Failed to await termination");
				e.printStackTrace();
			}
		}
		executor = null;
	}

}
