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

    Copyright David Picard - 2014

*/
package fr.lip6.jkernelmachines.classifier;

import static java.lang.Math.min;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import fr.lip6.jkernelmachines.density.DoubleKMeans;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.DebugPrinter;
import fr.lip6.jkernelmachines.util.algebra.MatrixVectorOperations;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

/**
 * <p>
 * Locally Linear SVM, as described in: <br />
 * Locally Linear Support Vector Machines<br />
 * L'ubor Ladicky, Philip H.S. Torr<br/>
 * Procedings of the 28th ICML, Bellevue, WA, USA, 2011.
 * </p>
 * 
 * @author picard
 *
 */
public class DoubleLLSVM implements Classifier<double[]> {
	
	DoubleKMeans km;
	double[][] W;
	double[] b;
	
	int K = 32;
	int E = 10;
	long t0 = 100;
	int skip = 10;
	double C = 1.0;
	int nn = 2;
	
	DebugPrinter debug = new DebugPrinter();

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#train(fr.lip6.jkernelmachines.type.TrainingSample)
	 */
	@Override
	public void train(TrainingSample<double[]> t) {
		throw new UnsupportedOperationException("Training on a single sample not supported at the moment");
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#train(java.util.List)
	 */
	@Override
	public void train(List<TrainingSample<double[]>> l) {
		
		nn = min(nn, K);

		// train gmm
		List<double[]> list = new ArrayList<>(l.size());
		for(TrainingSample<double[]> t : l) {
			list.add(t.sample);
		}
		km = new DoubleKMeans(K);
		km.train(list);
		debug.println(1, "KM trained");
		
		// compute likelihoods
		list.clear();
		List<Integer> index = new LinkedList<>();
		for(int i = 0 ; i < l.size() ; i++) {
			double[] d = likelihood(l.get(i).sample);
			list.add(d);
			index.add(i);
		}
		debug.println(1, "likelihood computed");
		
		// sgd on parameters
		W = new double[K][l.get(0).sample.length];
		b = new double[K];
		
		int count = skip;
		long t = t0;
		double lambda = 1./(C*l.size());
		for(int e = 0 ; e < E ; e++) {
			Collections.shuffle(index);
			for(int i : index) {
				double[] g = list.get(i);
				double[] x = l.get(i).sample;
				int y = l.get(i).label;
				
				// W
				double v = VectorOperations.dot(b, g);
				v += VectorOperations.dot(g, MatrixVectorOperations.rMul(W, x));
				if( 1 - y*v > 0) {
					double[][] grad = MatrixVectorOperations.outer(g, x);
					for(int m = 0 ; m < g.length ; m++) {
						for(int n = 0 ; n < x.length ; n++) {
							W[m][n] += 1./(lambda*t) * y * grad[m][n];
						}
					}
				}
				
				// b
				VectorOperations.addi(b, b, 1./(lambda*t)*y, g);
				
				if(--count < 0) {
					count = skip;
					for(int m = 0 ; m < g.length ; m++) {
						for(int n = 0 ; n < x.length ; n++) {
							W[m][n] *= 1 - skip/t;
						}
					}
				}
				
				t++;
			}

			// last regularization
			if(count > 0) {
				for(int m = 0 ; m < W.length ; m++) {
					for(int n = 0 ; n < W[m].length ; n++) {
						W[m][n] *= 1 - (skip-count)/t;
					}
				}
			}
			
			debug.println(1, "epoch "+e+" finished");
		}
		
		debug.println(2, "W: "+Arrays.deepToString(W));
		debug.println(2, "b: "+Arrays.toString(b));
		
	}
	
	private double[] likelihood(double[] e) {
		double[] d = km.distanceToMean(e);
		double[] ds = Arrays.copyOf(d, d.length);
		Arrays.sort(ds);
		double threshold = ds[nn-1]; 
		double sum = 0;
		for(int g = 0 ; g < K ; g++) {
			if(d[g] <= threshold) {
				d[g] = 1 / (1 + d[g]);
			}
			else {
				d[g] = 0;
			}
			sum += d[g];
		}
		// norm to unit sum
		VectorOperations.mul(d, 1./sum);
		return d;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(double[] e) {
		
		double[] d = likelihood(e);
		double out = VectorOperations.dot(d, MatrixVectorOperations.rMul(W, e));
		out += VectorOperations.dot(d, b);
		
		return out;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#copy()
	 */
	@Override
	public Classifier<double[]> copy() throws CloneNotSupportedException {
		return (DoubleLLSVM)this.clone();
	}

	/**
	 * return the number of anchor points
	 * @return the number of anchor points
	 */
	public int getK() {
		return K;
	}

	/**
	 * Sets the number of anchor points
	 * @param k the number of anchor points
	 */
	public void setK(int k) {
		K = k;
	}

	/**
	 * Returns the number of epochs for training
	 * @return the number of epochs
	 */
	public int getE() {
		return E;
	}

	/**
	 * Sets the number of epochs for training
	 * @param e the number of epochs
	 */
	public void setE(int e) {
		E = e;
	}

	/**
	 * Returns the hyperparameter C for the hinge loss tradeoff
	 * @return C
	 */
	public double getC() {
		return C;
	}

	/**
	 * Sets the hyperparameter C for the hinge loss tradeoff
	 * @param c
	 */
	public void setC(double c) {
		C = c;
	}

	/**
	 * Returns the number of anchor points taken into account by the model 
	 * @return the number of anchor points
	 */
	public int getNn() {
		return nn;
	}

	/**
	 * Sets the number of anchor points taken into account by the model
	 * @param nn the number of anchor opints
	 */
	public void setNn(int nn) {
		this.nn = nn;
	}

	/**
	 * Return the model hyperplanes
	 * @return W
	 */
	public double[][] getW() {
		return W;
	}

	/**
	 * Return the model biases
	 * @return b
	 */
	public double[] getB() {
		return b;
	}

}
