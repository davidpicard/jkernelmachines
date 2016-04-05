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
package net.jkernelmachines.classifier;

import java.util.ArrayList;
import java.util.List;

import net.jkernelmachines.kernel.Kernel;
import net.jkernelmachines.kernel.extra.NystromKernel;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.type.TrainingSampleStream;
import net.jkernelmachines.util.DebugPrinter;

/**
 * This classifier is a fast approximate SVM classifier using the Nystrom kernel
 * for project non-linearly the samples in a subspace in which a linear
 * classifier is learned.
 * 
 * @author picard
 * 
 */
public class NystromLSSVM<T> implements Classifier<T>, OnlineClassifier<T> {

	List<TrainingSample<T>> list;
	NystromKernel<T> kernel;
	DoubleSAG svm;

	double percent = 0.01;
	int iteration = 5;
	double C = 10;

	private DebugPrinter debug = new DebugPrinter();

	public NystromLSSVM(Kernel<T> k) {
		kernel = new NystromKernel<T>(k);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.classifier.Classifier#train(fr.lip6.jkernelmachines
	 * .type.TrainingSample)
	 */
	@Override
	public void train(TrainingSample<T> t) {
		if (list == null) {
			list = new ArrayList<TrainingSample<T>>();
		}
		list.add(t);
		train(list);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#train(java.util.List)
	 */
	@Override
	public void train(List<TrainingSample<T>> l) {

		list = new ArrayList<TrainingSample<T>>(l);
		int n = l.size();
		int m = (int) (n * percent) + 1;
		int p = m / iteration;
		if (p <= 0)
			p = 1;
		debug.println(3, "n = " + n + ", m = " + m + ", p = " + p);
		kernel.activeTrain(l, iteration, p, m);
		List<TrainingSample<double[]>> dl = kernel.projectList(l);
		svm = new DoubleSAG();
		svm.setLambda(1. / (C * n));
		svm.train(dl);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.classifier.OnlineClassifier#onlineTrain(fr.lip6
	 * .jkernelmachines.type.TrainingSampleStream)
	 */
	@Override
	public void onlineTrain(TrainingSampleStream<T> stream) {
		// TODO Auto-generated method stub

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.classifier.Classifier#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(T e) {
		double[] d = kernel.projectSample(e);
		return svm.valueOf(d);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#copy()
	 */
	@SuppressWarnings("unchecked")
	@Override
	public Classifier<T> copy() throws CloneNotSupportedException {
		return (NystromLSSVM<T>) this.clone();
	}

	/**
	 * Get the percentage of training set that is used to train the Nystrom
	 * approximation kernel
	 * 
	 * @return the percent of training set used
	 */
	public double getPercent() {
		return percent;
	}

	/**
	 * Set the percentage of training set to be used for training the Nystrom
	 * approximation kernel
	 * 
	 * @param percent
	 *            the percetage to use
	 */
	public void setPercent(double percent) {
		this.percent = percent;
	}

	/**
	 * Get the number of iterations used by the active learning strategy of the
	 * Nystrom kernel
	 * 
	 * @return the number of iterations
	 */
	public int getIteration() {
		return iteration;
	}

	/**
	 * Set the number of iteration to use in the Nystrom approximation
	 * 
	 * @param iteration
	 *            the number of iteration
	 */
	public void setIteration(int iteration) {
		this.iteration = iteration;
	}

	/**
	 * Get the C svm hyperparameter
	 * 
	 * @return C
	 */
	public double getC() {
		return C;
	}

	/**
	 * Set the C svm hyperparameter
	 * 
	 * @param c
	 *            Chyperparameter
	 */
	public void setC(double c) {
		C = c;
	}

}
