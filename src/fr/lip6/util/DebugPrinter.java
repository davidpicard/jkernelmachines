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

    Copyright David Picard - 2012

*/
package fr.lip6.util;

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
