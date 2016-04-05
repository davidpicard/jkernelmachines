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
