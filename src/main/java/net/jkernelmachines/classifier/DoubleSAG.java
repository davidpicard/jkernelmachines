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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.DebugPrinter;
import net.jkernelmachines.util.algebra.VectorOperations;

/**
 * Linear SVM using the SAG algorithm:
 * 
 * "A Stochastic Gradient Method with an Exponential Convergence Rate for
 * Strongly-Convex Optimization with Finite Training Sets",
 * Nicolas Le Roux, Mark Schmidt and Francis Bach.
 * 
 * @author picard
 * 
 */
public class DoubleSAG implements Classifier<double[]>, Serializable {

	private static final long serialVersionUID = -3497156096402090039L;
	// Available losses
	/** Type of loss function using hinge */
	public static final int HINGELOSS = 1;
	/** Type of loss function using a smoothed hinge */
	public static final int SMOOTHHINGELOSS = 2;
	/** Type of loss function using a squared hinge */
	public static final int SQUAREDHINGELOSS = 3;
	/** Type of loss function using log */
	public static final int LOGLOSS = 10;
	/** Type of loss function using margin log */
	public static final int LOGLOSSMARGIN = 11;

	// used loss function
	private int loss = HINGELOSS;
	// randomize indices to avoid perfect cycles (see Shalev-Schwartz 2013)
	private boolean cyclic = true;

	double[] w;
	double b = 0;
	double lambda = 1e-4;
	long E = 4;

	transient DebugPrinter debug = new DebugPrinter();

	// tmp variables
	private double alpha;
	private double[] yi;
	private double db;
	private double[] d;
	private int dim;
	private int n;


	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#train(java.util.List)
	 */
	@Override
	public void train(List<TrainingSample<double[]>> l) {
		dim = l.get(0).sample.length;
		n = l.size();

		yi = new double[l.size()];
		w = new double[dim];
		b = 0;

		// setting step size
		double L = 0;
		for (TrainingSample<double[]> t : l) {
			double norm = VectorOperations.n2(t.sample);
			if (norm > L)
				L = norm;
		}
		alpha = 1 / (4 * L);

		d = new double[dim];
		db = 0;

		// first epoch
		debug.println(3, "First epoch");
		for (int i = 0; i < l.size(); i++) {
			double[] x = l.get(i).sample;
			int y = l.get(i).label;

			// remove old gradient
			VectorOperations.addi(d, d, -yi[i] * y, x);
			db = db - yi[i] * y;

			// compute new derivative
			yi[i] = dloss(y * valueOf(x));

			// add new gradient
			VectorOperations.addi(d, d, yi[i] * y, x);
			db = db + yi[i] * y;

			VectorOperations.muli(w, w, 1 - alpha * lambda);
			VectorOperations.addi(w, w, alpha / (i + 1), d);
			b = (1 - alpha * lambda) * b + alpha * db / (i + 1);
		}

		// other epochs
		List<Integer> indices = new ArrayList<Integer>(n);
		for (int i = 0; i < n; i++)
			indices.add(i);

		for (int e = 0; e < E; e++) {
			debug.println(3, "epoch " + e);
			// randomizing indices to avoid perfect cycles (see Shalev-Schwartz
			// 2013)
			if (!cyclic)
				Collections.shuffle(indices);

			for (int ind = 0; ind < l.size(); ind++) {

				int i = ind;
				if (!cyclic)
					i = indices.get(ind);

				double[] x = l.get(i).sample;
				int y = l.get(i).label;

				update(i, x, y);
			}
		}

		debug.println(3, "w: " + Arrays.toString(w));
		debug.println(3, "b: " + b);

	}

	/**
	 * perform the single average gradient update on sample i (x, y)
	 */
	final private void update(int i, double[] x, int y) {
		// remove old gradient
		VectorOperations.addi(d, d, -yi[i] * y, x);
		db = db - yi[i] * y;

		// compute new derivative
		yi[i] = dloss(y * valueOf(x));

		// add new gradient
		VectorOperations.addi(d, d, yi[i] * y, x);
		db = db + yi[i] * y;

		VectorOperations.muli(w, w, 1 - alpha * lambda);
		VectorOperations.addi(w, w, alpha / n, d);
		b = (1 - alpha * lambda) * b + alpha * db / n;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.classifier.Classifier#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(double[] e) {
		return VectorOperations.dot(w, e);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#copy()
	 */
	@Override
	public Classifier<double[]> copy() throws CloneNotSupportedException {
		return (DoubleSAG) this.clone();
	}

	private double dloss(double z) {
		switch (loss) {
		case LOGLOSS:
			if (z < 0)
				return 1 / (Math.exp(z) + 1);
			double ez = Math.exp(-z);
			return ez / (ez + 1);
		case LOGLOSSMARGIN:
			if (z < 1)
				return 1 / (Math.exp(z - 1) + 1);
			ez = Math.exp(1 - z);
			return ez / (ez + 1);
		case SMOOTHHINGELOSS:
			if (z < 0)
				return 1;
			if (z < 1)
				return 1 - z;
			return 0;
		case SQUAREDHINGELOSS:
			if (z < 1)
				return (1 - z);
			return 0;
		default:
			if (z < 1)
				return 1;
			return 0;
		}
	}

	/**
	 * Tells the loss function the classifier is currently using
	 * 
	 * @return an integer specifying the loss function (HINGELOSS,
	 *         SQUAREDHINGELOSS, etc)
	 */
	public int getLoss() {
		return loss;
	}

	/**
	 * Sets the loss function to use for next training
	 * 
	 * @param loss
	 *            an integer specifying the loss to use (HINGELOSS,
	 *            SQUAREDHINGELOSS, etc)
	 */
	public void setLoss(int loss) {
		this.loss = loss;
	}

	/**
	 * Get the normal to the separating hyperplane
	 * 
	 * @return w, the hyperplane normal vector
	 */
	public double[] getW() {
		return w;
	}

	/**
	 * Set the normal to the hyperplane
	 * 
	 * @param w
	 *            the normal vector
	 */
	public void setW(double[] w) {
		this.w = w;
	}

	/**
	 * Get the bias of the classifier
	 * 
	 * @return the bias b
	 */
	public double getB() {
		return b;
	}

	/**
	 * Set the bias of the classifier
	 * 
	 * @param b
	 *            the bias
	 */
	public void setB(double b) {
		this.b = b;
	}

	/**
	 * Get the regularization parameter lambda
	 * 
	 * @return lambda the regularization parameter
	 */
	public double getLambda() {
		return lambda;
	}

	/**
	 * Set the regularization parameter lambda
	 * 
	 * @param lambda
	 *            the regularization parameter
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	/**
	 * Get the number of epochs (one pass through the entire data-set)
	 * 
	 * @return E number of epoch
	 */
	public long getE() {
		return E;
	}

	/**
	 * Set the number of epochs (one pass through the entire data-set)
	 * 
	 * @param e
	 *            the number of epoch
	 */
	public void setE(long e) {
		E = e;
	}

	/**
	 * Is the algorithm doing epoch of ordered samples
	 * 
	 * @return true if all epochs are through ordered samples, false if the
	 *         sample order is randomized at each epoch
	 */
	public boolean isCyclic() {
		return cyclic;
	}

	/**
	 * Set the order of the sample at each epoch
	 * 
	 * @param cyclic
	 *            true is the order remains the same through all epochs, false
	 *            is the order is randomized at each epoch
	 */
	public void setCyclic(boolean cyclic) {
		this.cyclic = cyclic;
	}

}
