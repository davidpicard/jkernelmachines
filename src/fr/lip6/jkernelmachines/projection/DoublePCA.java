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
package fr.lip6.jkernelmachines.projection;

import fr.lip6.jkernelmachines.threading.ThreadedMatrixOperator;
import fr.lip6.jkernelmachines.type.TrainingSample;
import static fr.lip6.jkernelmachines.util.algebra.MatrixOperations.eig;
import static fr.lip6.jkernelmachines.util.algebra.MatrixOperations.transi;
import static fr.lip6.jkernelmachines.util.algebra.VectorOperations.add;
import static fr.lip6.jkernelmachines.util.algebra.VectorOperations.dot;
import static fr.lip6.jkernelmachines.util.algebra.VectorOperations.mul;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Principal component analysis on double arrays.
 * @author picard
 *
 */
public class DoublePCA implements Serializable {
	
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
