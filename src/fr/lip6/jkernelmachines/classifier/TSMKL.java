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

    Copyright David Picard - 2012

*/
package fr.lip6.jkernelmachines.classifier;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import fr.lip6.jkernelmachines.kernel.Kernel;
import fr.lip6.jkernelmachines.kernel.adaptative.ThreadedSumKernel;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * <p>Implementation of the Two-Stages MKL solver by Abhishek Kumar et al.<br/>
 * This is a original implementation using the tools available in JKernelMachines, 
 * and not a Java conversion of the original matlab code.
 * </p>
 * 
 * <p>A Binary Classification Framework for Two-Stage Multiple Kernel Learning<br/>
 * Abhishek Kumar, Alexandru Niculescu-Mizil, Koray Kavukcoglu, Hal Daumé, <br/> 
 * ICML 2012. 
 * </p>
 * 
 * @author David Picard
 *
 */
public class TSMKL<T> implements KernelSVM<T> {
	
	
	LaSVM<T> lasvm;
	List<Kernel<T>> kernels;
	double[] beta;
	
	List<TrainingSample<T>> tlist;
	
	double lambda = 1.e-3;
	double C = 10;
	double t = 1;
	private DoubleLinear linear = new DoubleLinear();
	
	

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#train(fr.lip6.type.TrainingSample)
	 */
	@Override
	public void train(TrainingSample<T> t) {
		if(tlist == null)
			tlist = new ArrayList<TrainingSample<T>>();
		tlist.add(t);
		
		train(tlist);
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#train(java.util.List)
	 */
	@Override
	public void train(List<TrainingSample<T>> l) {
		
		tlist = new ArrayList<TrainingSample<T>>();
		tlist.addAll(l);
		
		// if no kernels bail out
		if(kernels == null || kernels.isEmpty())
			return;
		
		// 1. generate kernel matrices 
		List<double[][]> matrices = new ArrayList<double[][]>();
		for(Kernel<T> k : kernels)
			matrices.add(k.getKernelMatrix(tlist));
		
		// 2. generate samples in K-space		
		List<TrainingSample<double[]>> kslist = new ArrayList<TrainingSample<double[]>>();
		for(int i = 0 ; i < tlist.size() ; i++) {
			TrainingSample<T> xi = tlist.get(i);
			for(int j = 0 ; j < tlist.size() ; j++) {
				TrainingSample<T> xj = tlist.get(j);
			
				
				double[] zij = new double[kernels.size()];
				
				for(int m = 0 ; m < matrices.size() ; m++) {
					zij[m] = matrices.get(m)[i][j];
				}
				
				kslist.add(new TrainingSample<double[]>(zij, xi.label*xj.label));
			}
		}
		
		// 3. learn weights in K space
		Collections.shuffle(kslist);
		
		// crossval t0
		List<TrainingSample<double[]>> subkslist = kslist.subList(0, kslist.size()/2);
		double tMin = 1;
		double errMin = Double.MAX_VALUE;
		for(int t0 = 0 ; t0 < 9 ; t0++) {
			
			t = Math.pow(10, t0);
			
			//train
			init();
			trainOnce(subkslist);

			//error rate
			int nbErr = 0;
			for(TrainingSample<double[]> sam : kslist.subList(kslist.size()/2, kslist.size())) {
				if(sam.label * (linear.valueOf(sam.sample, beta))< 0)
					nbErr++;
			}
			if(nbErr < errMin) {
				tMin = t0;
				errMin = nbErr;
			}
		}
		t = Math.pow(10, tMin);
		
		// train svm in kspace to get weights
		init();		
		for(int e = 0 ; e < 5 ; e++)
			trainOnce(kslist);

		// 4. Profit!
		learnSVM();
		

	}
	
	// init the kernels weights
	private void init() {
		//new weights
		beta = new double[kernels.size()];
	}
	
	// train svm in kspace based on Leon Bottou's SGD
	private void trainOnce(List<TrainingSample<double[]>> kslist) {
		
		int imax = kslist.size();

		double wscale = 1;
		for (int i = 0; i < imax; i++) {
			double eta = 1.0 / (lambda * t);
			double s = 1 - eta * lambda;
			wscale *= s;
			if (wscale < 1e-9) {
				for (int d = 0; d < beta.length; d++)
					beta[d] *= wscale;
				wscale = 1;
			}
			double[] x = kslist.get(i).sample;
			double y = kslist.get(i).label;
			double v = linear.valueOf(x, beta);
			double z = y * v;

			if (z < 1) {
				for (int d = 0; d < beta.length; d++) {
					
					beta[d] += eta * x[d] * y / wscale;
					// allow only psd combinations to ensure the combined function is a Mercer kernel
					if(beta[d] < 0)
						beta[d] = 0.;
				}
			}
			t += 1;
		}
		
		for(int d = 0 ; d < beta.length ; d++)
			beta[d] *= wscale;
	}
	
	// learn kernel SVM using the weights from kspace
	private void learnSVM() {
		
		ThreadedSumKernel<T> tsk = new ThreadedSumKernel<T>();
		for(int k = 0 ; k < kernels.size() ; k++)
			if(beta[k] != 0)
				tsk.addKernel(kernels.get(k), beta[k]);
		lasvm = new LaSVM<T>(tsk);
		lasvm.setC(C);
		lasvm.train(tlist);
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(T e) {
		return lasvm.valueOf(e);
	}
	
	
	/**
	 * Adds a kernel to the combination
	 * @param k the kernel to add
	 */
	public void addKernel(Kernel<T> k) {
		if(kernels == null)
			kernels = new ArrayList<Kernel<T>>();
		kernels.add(k);
	}
	

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#copy()
	 */
	@SuppressWarnings("unchecked")
	@Override
	public TSMKL<T> copy() throws CloneNotSupportedException {
		return (TSMKL<T>)super.clone();
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.KernelSVM#setKernel(fr.lip6.kernel.Kernel)
	 */
	@Override
	public void setKernel(Kernel<T> k) {
		//do nothing
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.KernelSVM#getAlphas()
	 */
	@Override
	public double[] getAlphas() {
		return lasvm.getAlphas();
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.KernelSVM#setC(double)
	 */
	@Override
	public void setC(double c) {
		C = c;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.KernelSVM#getC()
	 */
	@Override
	public double getC() {
		return C;
	}
	
	/**
	 * tells the weights of the kernel combination
	 * @return the weights of the kernels
	 */
	public double[] getKernelWeights() {
		return beta;
	}

	/**
	 * Return the list of kernels used by this MKL algorithm
	 * @return kernels
	 */
	public List<Kernel<T>> getKernels() {
		return kernels;
	}

	
}
