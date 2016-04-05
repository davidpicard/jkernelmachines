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
package net.jkernelmachines.evaluation;

/**
 * @author picard
 *
 */
public interface MultipleEvaluatorCrossValidation<T> {
	
	/**
	 * Register a new evaluator to this corssvalidation
	 * @param name a string associated with this evaluator
	 * @param e the evaluator
	 */
	public void addEvaluator(String name, Evaluator<T> e);
	
	/**
	 * unregister the evaluator by its name
	 * @param name the name of the evaluator
	 */
	public void removeEvaluator(String name);
	
	/**
	 * Tells the average score of the test for the given evaluator
	 * @param name name of the evaluator
	 * @return the average score
	 */
	public double getAverageScore(String name);
	
	/**
	 * Tells the standard deviation of the test for the given evaluator
	 * @param name name of the evaluator
	 * @return the standard deviation
	 */
	public double getStdDevScore(String name);
	
	/**
	 * Tells the scores of the tests, in order of evaluation for the given evaluator
	 * @param name of the evaluator
	 * @return an array with the scores in order
	 */
	public double[] getScores(String name); 

}
