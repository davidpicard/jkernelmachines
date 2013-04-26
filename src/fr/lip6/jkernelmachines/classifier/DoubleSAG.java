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
package fr.lip6.jkernelmachines.classifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.DebugPrinter;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

/**
 * Linear SVM using the SAG algorithm:<br/>
 * 
 * "A Stochastic Gradient Method with an Exponential Convergence Rate for 
 * Strongly-Convex Optimization with Finite Training Sets", <br>
 * Nicolas Le Roux, Mark Schmidt and Francis Bach.
 * 
 * @author picard
 * 
 */
public class DoubleSAG implements Classifier<double[]> {

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

	double[] w;
	double b = 0;
	double lambda = 1e-4;
	long E = 2;

	DebugPrinter debug = new DebugPrinter();

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.classifier.Classifier#train(fr.lip6.jkernelmachines
	 * .type.TrainingSample)
	 */
	@Override
	public void train(TrainingSample<double[]> t) {
		throw new UnsupportedOperationException("Not applicable");

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#train(java.util.List)
	 */
	@Override
	public void train(List<TrainingSample<double[]>> l) {
		int dim = l.get(0).sample.length;
		int n = l.size();
		

		double[] yi = new double[l.size()];
		w = new double[dim];
		b = 0;

		// setting step size
		double L = 0;
		for (TrainingSample<double[]> t : l) {
			double norm = VectorOperations.n2(t.sample);
			if (norm > L)
				L = norm;
		}
		double alpha = 1/(2*L);

		double[] d = new double[dim];
		double db = 0;

		// first epoch
		debug.println(3, "First epoch");
		for (int i = 0; i < l.size(); i++) {
			double[] x = l.get(i).sample;
			int y = l.get(i).label;


			// remove old gradient
			for(int k = 0 ; k < dim ; k++){
				d[k] = d[k] - yi[i]*x[k]*y;
			}
			db = db - yi[i]*y;
			
			// compte new derivative
			yi[i] = dloss(y * (VectorOperations.dot(w, x)+b));
			
			// add new gradient
			for(int k = 0 ; k < dim ; k++) {
				d[k] = d[k] + yi[i]*x[k]*y;
			}
			db = db + yi[i]*y;

			for (int k = 0; k < w.length; k++) {
				w[k] = (1 - alpha * lambda) * w[k] + alpha * d[k] / (i+1);
			}
			b = (1 - alpha * lambda) * b + alpha * db / (i+1);
		}

		// other epochs
		List<Integer> indices = new ArrayList<Integer>(n);
		for(int i = 0 ; i < n ; i++)
			indices.add(i);
		
		for (int e = 0; e < E; e++) {
			debug.println(3, "epoch " + e);
			// randomizing indices to avoid perfect cycles (see Shalev-Schwartz 2013)
			Collections.shuffle(indices);

			for (int ind = 0; ind < l.size(); ind++) {
				int i = indices.get(ind);
				double[] x = l.get(i).sample;
				int y = l.get(i).label;


				// remove old gradient
				db = db - yi[i]*y;
				for(int k = 0 ; k < dim ; k++){
					d[k] = d[k] - yi[i]*x[k]*y;
				}
				
				// compute new derivative
				yi[i] = dloss(y * (VectorOperations.dot(w, x)+b));
				
				// add new gradient
				for(int k = 0 ; k < dim ; k++) {
					d[k] = d[k] + yi[i]*x[k]*y;
				}
				db = db + yi[i]*y;
				
				for (int k = 0; k < w.length; k++) {
					w[k] = (1 - alpha * lambda) * w[k] + alpha * d[k] / n;
				}
				b = (1 - alpha * lambda) * b + alpha * db / n;
			}
		}
		
		debug.println(3, "w: "+Arrays.toString(w));
		debug.println(3, "b: "+b);

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.classifier.Classifier#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(double[] e) {
		return VectorOperations.dot(w, e)+b;
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

	/** Tells the loss function the classifier is currently using
	 * 
	 * @return an integer specifying the loss function (HINGELOSS, SQUAREDHINGELOSS, etc)
	 */
	public int getLoss() {
		return loss;
	}

	/**
	 * Sets the loss function to use for next training
	 * @param loss an integer specifying the loss to use (HINGELOSS, SQUAREDHINGELOSS, etc)
	 */
	public void setLoss(int loss) {
		this.loss = loss;
	}

	
	public double[] getW() {
		return w;
	}

	public void setW(double[] w) {
		this.w = w;
	}

	public double getB() {
		return b;
	}

	public void setB(double b) {
		this.b = b;
	}

	public double getLambda() {
		return lambda;
	}

	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	public long getE() {
		return E;
	}

	public void setE(long e) {
		E = e;
	}

}
