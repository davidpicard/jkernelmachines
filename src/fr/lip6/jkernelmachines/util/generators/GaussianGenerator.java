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
package fr.lip6.jkernelmachines.util.generators;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * <p>
 * Class for generating toys subject to binary classification tasks, using
 * Gaussian distribution.
 * </p>
 * 
 * <p>
 * Toys are sampled from 2 Gaussian distributions (one for the positive samples,
 * the other for the negtive samples). The positive samples have a mean of p on
 * each component, while the negative samples have a mean of -p on each
 * component. The number of dimension, the spacing p and the standard deviation
 * of the Gaussian can be adjusted. <br />
 * Generated lists are shuffled.
 * </p>
 * 
 * @author picard
 * 
 */
public class GaussianGenerator {

	float p = 1.0f;
	double sigma = 1.0;
	int dimension = 3;

	Random ran = new Random();

	/**
	 * Default constructor: p = 1, sigma = 1.0 , dimension = 3.
	 */
	public GaussianGenerator() {
	}

	/**
	 * Constructor specifying dimension.
	 * 
	 * @param dimension
	 *            the dimension of the generated toys.
	 */
	public GaussianGenerator(int dimension) {
		this.dimension = dimension;
	}

	/**
	 * Constructor with all parameters
	 * 
	 * @param dimension
	 *            dimension of the toys
	 * @param p
	 *            half distance between the 2 classes
	 * @param sigma
	 *            standard deviation of the Gaussian
	 */
	public GaussianGenerator(int dimension, float p, double sigma) {
		this.dimension = dimension;
		this.p = p;
		this.sigma = sigma;
	}

	/**
	 * Generate a list of toys, with half of them being in the first class.
	 * 
	 * @param number
	 *            the number of toys to generate
	 * @return the list of toys
	 */
	public List<TrainingSample<double[]>> generateList(int number) {
		return generateList(number / 2, number / 2);
	}

	/**
	 * Generate a list of toys with specified number of positive samples, and
	 * negatives samples.
	 * 
	 * @param positives
	 *            the number of positive toys
	 * @param negatives
	 *            the number of negative toys
	 * @return the list of toys
	 */
	public List<TrainingSample<double[]>> generateList(int positives,
			int negatives) {
		List<TrainingSample<double[]>> list = new ArrayList<TrainingSample<double[]>>();

		// positives
		for (int i = 0; i < positives; i++) {
			double[] d = new double[dimension];
			for (int x = 0; x < dimension; x++) {
				d[x] = p + ran.nextGaussian() * sigma;
			}
			list.add(new TrainingSample<double[]>(d, 1));
		}
		// negatives
		for (int i = 0; i < negatives; i++) {
			double[] d = new double[dimension];
			for (int x = 0; x < dimension; x++) {
				d[x] = -p + ran.nextGaussian() * sigma;
			}
			list.add(new TrainingSample<double[]>(d, -1));
		}

		// shuffle
		Collections.shuffle(list, ran);
		return list;
	}

	/**
	 * Tells the distance between classes
	 * @return the p
	 */
	public float getP() {
		return p;
	}

	/**
	 * Sets the distance between classes
	 * @param p the p to set
	 */
	public void setP(float p) {
		this.p = p;
	}

	/**
	 * Tells the standard deviation of the toys
	 * @return the sigma
	 */
	public double getSigma() {
		return sigma;
	}

	/**
	 * Set the standard deviation of the toys
	 * @param sigma the sigma to set
	 */
	public void setSigma(double sigma) {
		this.sigma = sigma;
	}

	/**
	 * Tells the dimension of the toys
	 * @return the dimension
	 */
	public int getDimension() {
		return dimension;
	}

	/**
	 * Sets the dimension of the toys
	 * @param dimension the dimension to set
	 */
	public void setDimension(int dimension) {
		this.dimension = dimension;
	}

}
