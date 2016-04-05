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
package net.jkernelmachines.density;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import net.jkernelmachines.kernel.Kernel;
import net.jkernelmachines.util.DebugPrinter;

/**
 * Density function based on SMO algorithm.
 * 
 * @author dpicard
 * 
 * @param <T>
 *            Datatype of input space
 */
public class SMODensity<T> implements DensityFunction<T>, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 4738328902335184013L;

	private Kernel<T> K;
	private double[] alphas;
	// training set
	private ArrayList<T> set;

	private int size;

	DebugPrinter debug = new DebugPrinter();

	// parametres
	private final double epsilon = 0.001;
	private double C = 1;
	double tolerance = 1e-7;
	// cache d'erreur
	double cache[];

	/**
	 * Constructor using the specified kernel function for computing
	 * similarities among samples
	 * 
	 * @param K
	 *            the kernel to use
	 */
	public SMODensity(Kernel<T> K) {
		this.K = K;
	}

	@Override
	public void train(T e) {

		if (set == null) {
			set = new ArrayList<T>();
		}

		set.add(e);

		double[] a_tmp = Arrays.copyOf(alphas, alphas.length + 1);
		a_tmp[alphas.length] = 0.;
		alphas = a_tmp.clone();

		train();
	}

	@Override
	public void train(List<T> e) {
		if (set == null) {
			set = new ArrayList<T>();
		}

		for (T t : e)
			set.add(t);
		size = set.size();

		alphas = new double[size];
		Arrays.fill(alphas, 1./size);
//		alphas[0] = C / size;

		train();
	}

	// calcul de l'optimisation
	private void train() {
		cache = new double[size];
//		Arrays.fill(cache, -1.);
		for(int i = 0 ; i < size ; i++) {
			double z = 0;
			for(int j = 0 ; j < size ; j++)
				z += alphas[j]*K.valueOf(set.get(i), set.get(j));
			cache[i] = 1-z;
		}

		// C = 1. / size;

		int nChange = 0;
		boolean bExaminerTout = true;

		int ite = 0;
		// On examine les exemples, de préférence ceux qui ne sont pas au bords
		// (qui ne
		// sont pas des SV).

		while (nChange > 0 || bExaminerTout) {
			nChange = 0;
			if (bExaminerTout) {
				for (int i = 0; i < size; i++)
					if (examiner(i))
						nChange++;
			} else {
				for (int i = 0; i < size; i++)
					if (alphas[i] > epsilon && alphas[i] < C / size - epsilon)
						if (examiner(i))
							nChange++;
			}
			if (bExaminerTout)
				bExaminerTout = false;
			else if (nChange == 0)
				bExaminerTout = true;

			ite++;
			if (ite > 1000000) {
				debug.println(2, "Too many iterations...");
				break;
			}
		}

		debug.println(1, "trained in " + ite + " iterations.");

	}

	// look for violating pair to optimize
	private boolean examiner(int i1) {

		if (cache[i1] * alphas[i1] > epsilon
				|| cache[i1] * (alphas[i1] - C) > epsilon) {
			// if ((cache[i1] < -tolerance && alphas[i1] < C / size - epsilon)
			// || (cache[i1] > tolerance && alphas[i1] > epsilon)) {

			// most violating pair
			double rMax = 0;
			int i2 = alphas.length;
			for (int i = 0; i < alphas.length; i++)
				if (alphas[i] > epsilon && alphas[i] < C / size - epsilon) {
					double r = Math.abs(cache[i1] - cache[i]);
					if (r > rMax) {
						rMax = r;
						i2 = i;
					}
				}
			if (i2 < alphas.length)
				if (optimize(i1, i2)) {
					return true;
				}

			// look for a clearly violating pair
			int k0 = (new Random()).nextInt(alphas.length);
			for (int k = k0; k < k0 + alphas.length; k++) {
				i2 = k % alphas.length;
				if (alphas[i2] > epsilon && alphas[i2] < C / size - epsilon)
					if (optimize(i1, i2)) {
						return true;
					}
			}

			// take one randomly
			k0 = (new Random()).nextInt(size);
			for (int k = k0; k < k0 + size; k++) {
				i2 = k % size;
				if (optimize(i2, i1)) {
					return true;
				}
			}
		}

		// KKT condition ok, nothing more to do
		return false;
	}

	// minimal problem resolution
	boolean optimize(int i1, int i2) {
		if (i1 == i2)
			return false;

		int i;
		double delta = alphas[i1] + alphas[i2];

		double L, H;
		if (delta > C / size) {
			L = delta - C / size;
			H = C / size;
		} else {
			L = 0;
			H = delta;
		}

		if (L == H) {
			return false;
		}

		double k11 = K.valueOf(set.get(i1), set.get(i1));
		double k22 = K.valueOf(set.get(i2), set.get(i2));
		double k12 = K.valueOf(set.get(i1), set.get(i2));

		double a1, a2;
		double eta = 2 * k12 - k11 - k22;
		if (eta < 0) {
			a2 = alphas[i2] + (cache[i2] - cache[i1]) / eta;
			if (a2 < L)
				a2 = L;
			else if (a2 > H)
				a2 = H;
		} else {
			double c1 = eta / 2;
			double c2 = cache[i1] - cache[i2] - eta * alphas[i2];
			double Lp = c1 * L * L + c2 * L;
			double Hp = c1 * H * H + c2 * H;
			if (Lp > Hp + epsilon)
				a2 = L;
			else if (Lp < Hp + epsilon)
				a2 = H;
			else
				a2 = alphas[i2];
		}

		if (Math.abs(a2 - alphas[i2]) < epsilon * (a2 + alphas[i2] + epsilon)) {
			return false;
		}

		a1 = delta - a2;

		if (a1 < 0) {
			a2 += a1;
			a1 = 0;
		} else if (a1 > C / size) {
			a2 += a1 - C / size;
			a1 = C / size;
		}

		double t1 = a1 - alphas[i1];
		double t2 = a2 - alphas[i2];
		for (i = 0; i < alphas.length; i++)
			cache[i] += t1 * K.valueOf(set.get(i1), set.get(i)) + t2
					* K.valueOf(set.get(i2), set.get(i));

		alphas[i1] = a1;
		alphas[i2] = a2;

		return true;
	}

	@Override
	public double valueOf(T e) {

		double sum = 0.;
		for (int i = 0; i < size; i++)
			sum += alphas[i] * K.valueOf(e, set.get(i));

		return sum;
	}

	/**
	 * Tells the weights of the training samples
	 * 
	 * @return an array of double representing the weights in the training list
	 *         order
	 */
	public double[] getAlphas() {
		return alphas;
	}

	/**
	 * Tells the hyperparameter C
	 * 
	 * @return C
	 */
	public double getC() {
		return C;
	}

	/**
	 * Sets the hyperparameter C
	 * 
	 * @param c
	 *            the hyperparameter C
	 */
	public void setC(double c) {
		C = c;
	}

}
