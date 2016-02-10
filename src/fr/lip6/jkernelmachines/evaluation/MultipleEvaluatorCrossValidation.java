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
package fr.lip6.jkernelmachines.evaluation;

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
