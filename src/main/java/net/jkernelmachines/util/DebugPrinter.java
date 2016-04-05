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
package net.jkernelmachines.util;

/**
 * Very basic library wide debug utility class.
 * @author picard
 *
 */
public final class DebugPrinter {
	
	/**
	 * level of debug information to print: 0 = none, 1 = some, 2 = more, ...
	 */
	public static int DEBUG_LEVEL = 0;
	
	/**
	 * Set level of debug information to print
	 * @param debug level of verbosity
	 */
	public static void setDebugLevel(int debug) {
		DEBUG_LEVEL = debug;
	}
	
	/**
	 * Println object to standard error stream iff debug level is 
	 * more than debug argument
	 * @param debug level at which to print the message
	 * @param o the message to print
	 */
	public final void println(int debug, Object o) {
		if(DEBUG_LEVEL >= debug) {
			System.err.println(o);
		}
	}

	
	/**
	 * Print object to standard error stream iff debug level is 
	 * more than debug argument
	 * @param debug level at which to print the message
	 * @param o the message to print
	 */
	public void print(int debug, Object o) {
		if(DEBUG_LEVEL >= debug) {
			System.err.print(o);
		}
	}

}
