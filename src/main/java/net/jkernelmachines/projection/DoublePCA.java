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
package net.jkernelmachines.projection;

import static net.jkernelmachines.util.algebra.MatrixOperations.eig;
import static net.jkernelmachines.util.algebra.MatrixOperations.transi;
import static net.jkernelmachines.util.algebra.VectorOperations.add;
import static net.jkernelmachines.util.algebra.VectorOperations.dot;
import static net.jkernelmachines.util.algebra.VectorOperations.mul;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import net.jkernelmachines.threading.ThreadedMatrixOperator;
import net.jkernelmachines.type.TrainingSample;

/**
 * Principal component analysis on double arrays.
 * @author picard
 *
 */
public class DoublePCA implements Serializable {
	
	private static final long serialVersionUID = -6200080076438113052L;
	double[][] projectors;
	double[] whitening_coeff;
	double[] mean;
	
	/**
	 * Train the projectors on a given data-set.
	 * @param list the list of training samples
	 */
	public void train(final List<TrainingSample<double[]>> list) {
		
		int dim = list.get(0).sample.length;
		
		mean  = new double[dim];
		for(TrainingSample<double[]> t : list) {
			mean = add(mean, 1, t.sample);
		}
		mean = mul(mean, 1./list.size());
		
		// compute covariance matrix;
		double[][] cov = new double[dim][dim];
		
		ThreadedMatrixOperator factory = new ThreadedMatrixOperator() {
			
			@Override
			public void doLines(double[][] matrix, int from, int to) {
				for(int i = from ; i < to ; i++) {
					for(int j = 0 ; j < matrix.length ; j++) {
						double sum = 0;
						for(TrainingSample<double[]> t : list) {
							sum += (t.sample[i]-mean[i]) * (t.sample[j]-mean[j]);
						}
						matrix[i][j] = sum/list.size();
					}
				}				
			}
		};
		
		cov = factory.getMatrix(cov);
		
		// eigen decomposition
		double[][][] eig = eig(cov);
		
		//projectors are eigenvectors transposed
		projectors = transi(eig[0]);
		
		//coefficients are the square root of the eigenvalues
		whitening_coeff = new double[dim];
		for(int d = 0 ; d < dim ; d++) {
			if(eig[1][d][d] > 0) {
				whitening_coeff[d] = 1./Math.sqrt(eig[1][d][d]);
			}
			else {
				whitening_coeff[d] = 0;
			}
		}
	}
	
	/**
	 * Project a single sample using the trained projectors.
	 * @param s the sample to project
	 * @return a new sample with the projected vector, and the same label
	 */
	public TrainingSample<double[]> project(TrainingSample<double[]> s) {
		double[] px = new double[projectors.length];
		
		for(int i = 0 ; i < projectors.length ; i++) {
			px[i] = dot(projectors[i], add(s.sample, -1, mean));
		}
		
		return new TrainingSample<double[]>(px, s.label);
	}
	
	/**
	 * Project a single sample using the trained projectors with optional whitening (unitary 
	 * covariance matrix).
	 * @param s the sample to project
	 * @param whitening option to perform a whitened projection
	 * @return a new sample with the projected vector and the same label
	 */
	public TrainingSample<double[]> project(TrainingSample<double[]> s, boolean whitening) {
		double[] px = new double[projectors.length];
		
		for(int i = 0 ; i < projectors.length ; i++) {
			px[i] = dot(projectors[i], add(s.sample, -1, mean));
			if(whitening) {
				px[i] *= whitening_coeff[i];
			}
		}
		
		return new TrainingSample<double[]>(px, s.label);
	} 
	
	/**
	 * Performs the projection on a list of samples.
	 * @param list the list of input samples
	 * @return a new list with projected samples
	 */
	public List<TrainingSample<double[]>> projectList(final List<TrainingSample<double[]>> list) {
		List<TrainingSample<double[]>> out = new ArrayList<TrainingSample<double[]>>();
		
		for(TrainingSample<double[]> t : list) {
			out.add(project(t));
		}
		
		return out;
	}

	/**
	 * Performs the projection on a list of samples with optional whitening (unitary covariance matrix).
	 * @param list the list of input samples
	 * @param whitening option to perform a whitened projection
	 * @return a new list with projected samples
	 */
	public List<TrainingSample<double[]>> projectList(final List<TrainingSample<double[]>> list, boolean whitening) {
		List<TrainingSample<double[]>> out = new ArrayList<TrainingSample<double[]>>();
		
		for(TrainingSample<double[]> t : list) {
			out.add(project(t, whitening));
		}
		
		return out;
	}

}
