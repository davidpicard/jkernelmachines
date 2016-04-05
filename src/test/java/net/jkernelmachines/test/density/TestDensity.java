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
package net.jkernelmachines.test.density;

import java.util.ArrayList;
import java.util.Random;

import net.jkernelmachines.density.DoubleGaussianMixtureModel;
import net.jkernelmachines.density.ParzenDensity;
import net.jkernelmachines.density.SDCADensity;
import net.jkernelmachines.density.SMODensity;
import net.jkernelmachines.density.SimpleMKLDensity;
import net.jkernelmachines.kernel.typed.DoubleGaussL2;
import net.jkernelmachines.kernel.typed.index.IndexDoubleGaussL2;

/**
 * Test cases of density estimators using generated data.
 * 
 * @author picard
 * 
 */
@Deprecated
public class TestDensity {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		int dimension = 3;
		int nbPosTrain = 250;
		int nbPosTest = 25;

		Random ran = new Random(System.currentTimeMillis());

		ArrayList<double[]> train = new ArrayList<double[]>();
		// 1. generate positive train samples
		for (int i = 0; i < nbPosTrain; i++) {
			double[] t = new double[dimension];
			for (int x = 0; x < dimension; x++) {
				t[x] = ran.nextGaussian();
			}

			train.add(t);
		}
		System.out.println("Samples generated");

		// 3. train svm
		long time = System.currentTimeMillis();
		// Kernel<double[]> k = new DoubleLinear();
		DoubleGaussL2 k = new DoubleGaussL2();
		k.setGamma(2);
		SMODensity<double[]> svm = new SMODensity<double[]>(k);
		svm.setC(100);
		svm.train(train);
		long smotime = System.currentTimeMillis() - time;
		System.out.println("SMO done in " + smotime);

		ParzenDensity<double[]> parzen = new ParzenDensity<double[]>(k);
		parzen.train(train);

		SDCADensity<double[]> sdca = new SDCADensity<double[]>(k);
		sdca.setC(100);
		sdca.train(train);

		DoubleGaussianMixtureModel gmm = new DoubleGaussianMixtureModel(1);
		gmm.train(train);

		SimpleMKLDensity<double[]> mkl = new SimpleMKLDensity<double[]>();
		for (int x = 0; x < dimension; x++) {
			mkl.addKernel(new IndexDoubleGaussL2(x));
		}
		mkl.setC(100);
		mkl.train(train);

		ArrayList<double[]> test = new ArrayList<double[]>();
		// 4. generate positive test samples
		for (int i = 0; i < nbPosTest; i++) {
			double[] t = new double[dimension];
			for (int x = 0; x < dimension; x++) {
				t[x] = ran.nextGaussian();
			}

			test.add(t);
		}

		// 6. test svm
		for (double[] t : test) {
			double value = svm.valueOf(t);
			double pvalue = parzen.valueOf(t);
			double dvalue = sdca.valueOf(t);
			double gvalue = gmm.valueOf(t);
			double mvalue = mkl.valueOf(t);

			System.out.println("smo: " + value + ", parzen: "
					+ pvalue + ", sdca: " + dvalue + " , gmm: "
					+ gvalue + " , mkl: " + mvalue);

		}

		double[] alphas = svm.getAlphas();
		int nnz = 0;
		for (double d : alphas)
			if (d > 0)
				nnz++;

		System.out.println("Non zeros coefficients : " + nnz + " ("
				+ (nnz / (double) alphas.length) + ")");

	}

}
