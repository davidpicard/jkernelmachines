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
