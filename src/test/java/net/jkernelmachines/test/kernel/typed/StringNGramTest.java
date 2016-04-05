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

    Copyright David Picard - 2014

*/
package net.jkernelmachines.test.kernel.typed;

import static org.junit.Assert.*;
import net.jkernelmachines.kernel.typed.StringNGram;

import org.junit.Test;

/**
 * @author picard
 *
 */
public class StringNGramTest {

	/**
	 * Test method for {@link net.jkernelmachines.kernel.typed.StringNGram#valueOf(java.lang.String, java.lang.String)}.
	 */
	@Test
	public final void testValueOfStringString() {

		String s1, s2;
		
		StringNGram k = new StringNGram(2);
		
		s1 = "ababab";
		
		s2 = "abab";
		assertEquals(8, k.valueOf(s1, s2), 1e-15);
		
		s2 = "ab sdlfijcfgh dfgiljd fgoidjfgdfigj dofigjdfg ab sdiofgj dfgoijdfgd fogidjfg ";
		assertEquals(6, k.valueOf(s1, s2), 1e-15);
		
		
		k = new StringNGram(3);
		s1 = "ababab";
		s2 = "abab";
		assertEquals(4, k.valueOf(s1, s2), 1e-15);
		
		s2 = "ab sdlfijcfgh dfgiljd fgoidjfgdfigj dofigjdfg ab sdiofgj dfgoijdfgd fogidjfg ";
		assertEquals(0, k.valueOf(s1, s2), 1e-15);
		
		
	}

}
