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
package net.jkernelmachines.util.generators;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import net.jkernelmachines.type.TrainingSample;

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
 * of the Gaussian can be adjusted.
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
