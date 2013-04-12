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

import java.util.ArrayList;
import java.util.List;

import fr.lip6.jkernelmachines.kernel.Kernel;
import fr.lip6.jkernelmachines.type.TrainingSample;
import static fr.lip6.jkernelmachines.util.algebra.MatrixOperations.eig;
import static fr.lip6.jkernelmachines.util.algebra.MatrixOperations.transi;

/**
 * @author picard
 *
 */
public class KernelPCA<T> {
	
	private Kernel<T> kernel;
	private double[][] projectors;
	private double[] whiteningCoefficients;
	private List<TrainingSample<T>> list;
	private int dim;
	
	public KernelPCA(Kernel<T> k) {
		this.kernel = k;
	}
	
	public void train(List<TrainingSample<T>> list) {
		this.list = list;
		
		// SVD of kernel matrix
		double[][] K = kernel.getKernelMatrix(list);
		double[][][] eig = eig(K);
		dim = eig[0].length;
		
		// projectors
		projectors = transi(eig[0]);
		
		// whitening coeff
		whiteningCoefficients = new double[dim];
		for(int d = 0 ; d < dim ; d++) {
			double p = eig[1][d][d];
			if(p > 0) {
				whiteningCoefficients[d] = 1./Math.sqrt(p);
			}
		}
	}
	
	public TrainingSample<double[]> project(TrainingSample<T> t, boolean whitening) {
		double[] proj = new double[dim];
		
		for(int d = 0 ; d < dim ; d++){
			for(int i = 0 ; i < list.size() ; i++) {
				proj[d] += projectors[d][i] * kernel.valueOf(list.get(i).sample, t.sample);
			}
			proj[d] *= whiteningCoefficients[d];
			if(whitening) {
				proj[d] *= whiteningCoefficients[d];
			}
		}
		
		return new TrainingSample<double[]>(proj, t.label);
	}
	
	public TrainingSample<double[]> project(TrainingSample<T> t) {
		return project(t, false);
	}
	
	/**
	 * Performs the projection on a list of samples.
	 * @param list the list of input samples
	 * @return a new list with projected samples
	 */
	public List<TrainingSample<double[]>> projectList(final List<TrainingSample<T>> list) {
		List<TrainingSample<double[]>> out = new ArrayList<TrainingSample<double[]>>();
		
		for(TrainingSample<T> t : list) {
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
	public List<TrainingSample<double[]>> projectList(final List<TrainingSample<T>> list, boolean whitening) {
		List<TrainingSample<double[]>> out = new ArrayList<TrainingSample<double[]>>();
		
		for(TrainingSample<T> t : list) {
			out.add(project(t, whitening));
		}
		
		return out;
	}

}
