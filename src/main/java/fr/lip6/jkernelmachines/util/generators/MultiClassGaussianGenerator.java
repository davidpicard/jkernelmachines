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
