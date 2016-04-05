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
 * Class for generating toys subject to multi-class classification tasks, using
 * Gaussian distributions.
 * </p>
 * 
 * <p>
 * Toys are sampled from as many Gaussian distributions as the number of
 * classes, in a space with the dimension of the number of classes. Each
 * Gaussian has a mean of 1 on the component of the class and 0 otherwise. Thee
 * spacing p and the standard deviation of the Gaussian can be adjusted.
 * Generated lists are shuffled.
 * </p>
 * 
 * @author picard
 * 
 */
public class MultiClassGaussianGenerator {

	float p = 2.0f;
	double sigma = 1.0;
	int nbclasses = 5;

	Random ran = new Random();

	/**
	 * Default constructor, with p=2, sigma = 1, nbclasses = 5;
	 */
	public MultiClassGaussianGenerator() {
	}

	/**
	 * Constructor specifying the number of classes
	 * 
	 * @param nbclasses
	 *            the number of classes
	 */
	public MultiClassGaussianGenerator(int nbclasses) {
		this.nbclasses = nbclasses;
	}

	/**
	 * Generates a list of Toys with specified number of samples per class
	 * 
	 * @param samplesPerClass
	 *            the number of samples for each class
	 * @return the list of samples
	 */
	public List<TrainingSample<double[]>> generateList(int samplesPerClass) {
		List<TrainingSample<double[]>> list = new ArrayList<TrainingSample<double[]>>();

		for (int c = 0; c < nbclasses; c++) {
			for (int i = 0; i < samplesPerClass; i++) {
				double[] d = new double[nbclasses];

				for (int x = 0; x < nbclasses; x++) {
					if (c == x)
						d[x] = p + ran.nextGaussian()*sigma;
					else
						d[x] = ran.nextGaussian()*sigma;
				}

				list.add(new TrainingSample<double[]>(d, c));

			}
		}

		// shuffle the list before returning
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
	 * Tells the standard deviation 
	 * @return the sigma
	 */
	public double getSigma() {
		return sigma;
	}

	/**
	 * Sets the standard deviation
	 * @param sigma the sigma to set
	 */
	public void setSigma(double sigma) {
		this.sigma = sigma;
	}

	/**
	 * Tells the number of classes
	 * @return the nbclasses
	 */
	public int getNbclasses() {
		return nbclasses;
	}

	/**
	 * Sets the number of classes
	 * @param nbclasses the nbclasses to set
	 */
	public void setNbclasses(int nbclasses) {
		this.nbclasses = nbclasses;
	}

}
