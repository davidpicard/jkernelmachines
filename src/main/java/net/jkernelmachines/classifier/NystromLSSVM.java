package net.jkernelmachines.classifier;
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

    Copyright David Picard - 2013

 */

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
	
	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.classifier.OnlineClassifier#onlineTrain(fr.lip6.jkernelmachines.type.TrainingSampleStream)
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
