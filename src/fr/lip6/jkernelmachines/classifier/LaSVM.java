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

    Copyright David Picard - 2010

*/
/**
 * 
 */
package fr.lip6.jkernelmachines.classifier;

import fr.lip6.jkernelmachines.kernel.Kernel;
import fr.lip6.jkernelmachines.threading.ThreadPoolServer;
import fr.lip6.jkernelmachines.threading.ThreadedMatrixOperator;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.DebugPrinter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

/**
 * <p>
 * Kernel SVM classifier implementing the LaSVM algorithm
 * </p>
 * <p>
 * <b>Fast Kernel Classifiers with Online and Active Learning</b><br />
 * Antoine Bordes, Seyda Ertekin, Jason Weston and Léon Bottou<br />
 * <i>Journal of Machine Learning Research, 6:1579-1619, September 2005. </i>
 * </p>
 * @author picard
 *
 */
public final class LaSVM<T> implements KernelSVM<T>, Serializable {

	private static final long serialVersionUID = -831288193185967121L;

	private Kernel<T> kernel;
	
	private List<TrainingSample<T>> tlist; //training set
	private T[] tarray;
	
	private boolean[] S;
	private double[] alphas;
	private int[] y;
	private double[] g;//gradient on support vectors
	private double[] Cmin, Cmax; // minmax for support vectors
	private int imin, imax;
	private double gmin, gmax;
	private double[][] kmatrix;
	private boolean minmaxFlag = false;
	private double[] kmaxmin;
	private final LinkedList<Integer> trainQueue = new LinkedList<Integer>();
	
	private double b = 0; // bias
	
	private double C = 1.0; //hyperparameter C
	private int E = 5; //number of epochs
	private static final double tau = 1e-15;
	
	private static final int initSampling = 5;
	
	transient DebugPrinter debug = new DebugPrinter();
	
	/**
	 * Constructor with specific kernel
	 * @param k the kernel used by this classifier
	 */
	public LaSVM(Kernel<T> k)
	{
		this.kernel = k;
	}
	
	
	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#train(fr.lip6.type.TrainingSample)
	 */
	@Override
	public void train(TrainingSample<T> t) {
		if(tlist == null) {
			tlist = new ArrayList<TrainingSample<T>>();
			tlist.add(t);
			init();
		}
		else {
			tlist.add(t);
			//compute kernel
			kmatrix = kernel.getKernelMatrix(tlist);
			
			
			int idx = tlist.size()-1;
			// update variables
			S = Arrays.copyOf(S, tlist.size());
			S[idx] = true;
			alphas = Arrays.copyOf(alphas, tlist.size());
			y = Arrays.copyOf(y, tlist.size());
			y[idx] = t.label;
			
			g = Arrays.copyOf(g, tlist.size());
			g[idx] =  y[idx];
			for(int s = 0 ; s < alphas.length ; s++)
				g[idx] -= alphas[s] * kmatrix[idx][s];
			kmaxmin = Arrays.copyOf(kmaxmin, tlist.size());
			
			imin = -1;
			imax = -1;
			minmaxFlag = false;
			
			//set minmax
			Cmin = Arrays.copyOf(Cmin, tlist.size());
			Cmax = Arrays.copyOf(Cmax, tlist.size());
			Cmin[y.length-1] = Math.min(C*y[y.length-1] , 0);
			Cmax[y.length-1] = Math.max(C*y[y.length-1], 0);
			
			
		}
		train();
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#train(java.util.List)
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void train(List<TrainingSample<T>> l) {
		tlist = new ArrayList<TrainingSample<T>>();
		tlist.addAll(l);
		
		if(tlist.isEmpty()) {
			return;
		}
		
		// handle single label
		int yref = tlist.get(0).label;
		boolean single = true;
		for(TrainingSample<T> t  : tlist) {
			if(t.label != yref) {
				single = false;
			}
		}
		if(single) {
			// default to parzen
			tarray = (T[]) new Object[tlist.size()];
			for(int i = 0 ; i < tarray.length ; i++)
				tarray[i] = tlist.get(i).sample;
			
			alphas = new double[tlist.size()];
			Arrays.fill(alphas, yref*C);
			
			S = new boolean[tlist.size()];
			Arrays.fill(S, true);
			
			y = new int[tlist.size()];
			Arrays.fill(y, yref);
					
			return;
		}

		//init S and gs
		init();
		
		train();
	}
	
	
	@SuppressWarnings("unchecked")
	private void train()
	{

		trainQueue.clear();
		for(int e = 0 ; e < E ; e++)
		{
			//do one pass over the entire (shuffled) training set
			for(int i = 0 ; i < tlist.size() ; i++)
				trainQueue.add(i);
//			Collections.shuffle(trainQueue);
			while(!trainQueue.isEmpty())
			{
				//step 2.
				process(trainQueue.poll());
				//	step 3
//				while(optim(-1, -1));
				reprocess();
			}
		}
		// step 4
		int iter = 100000;
		while(optim(-1, -1) && iter-- > 0);
		if(iter == 0)
			debug.println(2, "*** lasvm : too much reprocess.");
		reprocess();
		
		//clear non sv
		for(int s = 0 ; s < S.length ; s++)
		{
			if(alphas[s] == 0)
				S[s] = false;
			if(!S[s])
				alphas[s] = 0.;
		}

		//update b and delta
		minmax();
		b = (gmax + gmin) /2;
		
		
		tarray = (T[]) new Object[tlist.size()];
		for(int i = 0 ; i < tarray.length ; i++)
			tarray[i] = tlist.get(i).sample;
		
		//clear cache
		kmatrix = null;

		//stop threads
		ThreadPoolServer.shutdownNow();
	}
	
	public void retrain()
	{
		//rebuild matrix
		kmatrix = kernel.getKernelMatrix(tlist);
		
		//rebuild gradient
		final double[] tmp = new double[g.length];
		(new ThreadedMatrixOperator(){

			@Override
			public void doLines(double[][] matrix, int from, int to) {
				for(int index = from ; index < to ; index++)
				{
					if(S[index])
					{
						tmp[index] = y[index];
						for(int s = 0 ; s < matrix[index].length ; s++)
							tmp[index] -= alphas[s] * kmatrix[index][s];
					}
				}
			}}).getMatrix(kmatrix);
		g = tmp;
	
		// refresh all sv
		// step 4
		int iter = 100000;
		while(optim(-1, -1) && iter-- > 0);
		if(iter == 0)
			debug.println(2, "*** lasvm : too much reprocess.");
		reprocess();
		

		//stop threads
		ThreadPoolServer.shutdownNow();
	}
	
	//init by adding a few samples to S
	private void init()
	{
		S = new boolean[tlist.size()];
		Arrays.fill(S, false);
		alphas = new double[tlist.size()];
		y = new int[tlist.size()];
		g = new double[tlist.size()];
		kmaxmin = new double[tlist.size()];
		
		imin = -1;
		imax = -1;
		minmaxFlag = false;
		
		//set minmax
		Cmin = new double[tlist.size()];
		Cmax = new double[tlist.size()];
		for(int i = 0 ; i < Cmin.length ; i++)
		{
			y[i] = tlist.get(i).label;
			Cmin[i] = Math.min(C*y[i] , 0);
			Cmax[i] = Math.max(C*y[i], 0);
		}
		
		//compute kernel
		kmatrix = kernel.getKernelMatrix(tlist);
		
		//add at most min(initSampling, maxpos) positives examples
		//and min(initSampling, maxneg) negatives samples
		int nbpos = 0 , nbneg = 0;
		for(int i = 0 ; i < alphas.length ; i++)
		{
			if(y[i] == 1 && nbpos < initSampling)
			{
				S[i] = true;
				g[i] = 1; //initial gradient is y
				nbpos++;
			}
			if(y[i] == -1 && nbneg < initSampling)
			{
				S[i] = true;
				g[i] = -1; //initial gradient is y
				nbneg++;
			}
			if(nbpos > initSampling && nbneg > initSampling)
				break;
		}
		
	}
	
	private final void minmax()
	{
		if(minmaxFlag)
			return;
		imin = -1;
		imax = -1;
		gmin = Double.MAX_VALUE;
		gmax = -Double.MAX_VALUE;
		for(int s =0 ; s < S.length ; s++)
		{
			if(!S[s])
				continue;
			double as = alphas[s];
			double gs = g[s];
			if(as > Cmin[s] && gs < gmin)
			{
				gmin = gs;
				imin = s;
			}
			if(as < Cmax[s] && gs > gmax)
			{
				gmax = gs;
				imax = s;
			}
		}
		minmaxFlag = true;
	}
	
	private boolean optim(int imin, int imax)
	{
		minmax();
		if(imin < 0)
			imin = this.imin;
		if(imax < 0)
			imax = this.imax;
		
		if(imin < 0 || imax < 0)
			return false;
		
		gmin = g[imin];
		gmax = g[imax];
		double gmaxmin = g[imax] - g[imin];
		
		if(gmaxmin < tau)
			return false;
		
		//max admissible step
		double step = Math.min(alphas[imin] - Cmin[imin], Cmax[imax] - alphas[imax]);
		if(step == 0)
			return false;
		
		double kminmin = kmatrix[imin][imin];
		double kmaxmax = kmatrix[imax][imax];
		double kminmax = kmatrix[imin][imax];
		step = Math.min((gmaxmin)/(kminmin+kmaxmax-2*kminmax), step);
		//update
		alphas[imax] += step;
		alphas[imin] -= step;
				
		for(int s = 0 ; s < S.length ; s++)
			kmaxmin[s] = kmatrix[imax][s] - kmatrix[imin][s];
		
		for(int s = 0 ; s < S.length ; s++)
		{
			if(S[s])
			{
				g[s] -= step * kmaxmin[s];
			}
		}
		
		minmaxFlag = false;
		return true;
	}
	
	private boolean process(int k)
	{
		//bail out if S contains t
		if(S[k])
		{
			return false;
		}
		
		//bail out if non labeled example
		if(y[k] != 1 && y[k] != -1)
		{
			return false;
		}
		
		//compute gradient
		alphas[k] = 0;
		double gk = y[k];
		for(int s = 0 ; s < S.length ; s++)
		{
			if(!S[s])
				continue;
			gk -= alphas[s] * kmatrix[k][s];
		}
		//decide insertion
		minmax();
		if(gmin < gmax)
		if( (Cmin[k] >= 0 && gk< gmin) || ( Cmax[k] <= 0 && gk > gmax))
			return false;
		
		//insert
		S[k] = true;
		g[k] = gk;
		minmaxFlag = false; // we changed g
		
		//find violating pair
		if(Cmin[k] >= 0)
		{
			optim(-1, k);
		}
		else
		{
			optim(k, -1);
		}		
		return true;
	}
	
	private boolean reprocess()
	{
		//optim violating pair
		boolean status = optim(-1, -1);
		
		//reduce expansion
		//4. research for gmax and gmin
		minmax();
		
		//prune S
		for(int s = 0 ; s < S.length ; s++)
		{
			if(S[s] && alphas[s] == 0)
			if(y[s] == -1)
			{
				if( g[s] >= gmax)
					S[s] = false;
			}
			else if(g[s] <= gmin)
			{
				S[s] = false;
			}
		}
		
		return status;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#valueOf(java.lang.Object)
	 */
	@Override
	public final double valueOf(T e) {
		if(S == null) {
			return 0;
		}
		double r = b;
		//cache kernel
		double[] kline = new double[S.length];
		for(int s = 0 ; s < alphas.length ; s++)
			if(S[s])
				kline[s] = kernel.valueOf(tarray[s], e);
		//do product
		for(int s = 0 ; s < S.length ; s++)
			if(S[s])
				r += alphas[s] * kline[s];
		return r;
	}

	/**
	 * Tells the hyperparamter C 
	 * @return C
	 */
	public double getC() {
		return C;
	}


	/**
	 * Sets the hyperparameter C
	 * @param c the hyperparameter C
	 */
	public void setC(double c) {
		C = c;
	}


	/**
	 * Tells the number of epochs used for training
	 * @return the number of epochs
	 */
	public int getE() {
		return E;
	}


	/**
	 * Sets the number of epochs used for training
	 * @param e the number of epochs
	 */
	public void setE(int e) {
		E = e;
	}

	/**
	 * Tells the numerical precision 
	 * @return the numerical precision
	 */
	public double getTau() {
		return tau;
	}

	/**
	 * Tells the array of support vector coefficients
	 * @return the array of support vector coefficients
	 */
	public double[] getAlphas() {
		double[] a = new double[alphas.length];
		for(int s = 0 ; s < a.length ; s++)
			a[s] = alphas[s] * y[s];
		return a;
	}

	/**
	 * Tells the bias term
	 * @return the bias term
	 */
	public double getB() {
		return b;
	}

	/**
	 * Sets the bias term
	 * @param b the new bias term
	 */
	public void setB(double b) {
		this.b = b;
	}

	/**
	 * Tells the kernel used by this classifier
	 * @return the kernel used by this classifier
	 */
	public Kernel<T> getKernel() {
		return kernel;
	}

	/**
	 * Sets the kernel used by this classifier
	 * @param kernel the kernel to use
	 */
	public void setKernel(Kernel<T> kernel) {
		this.kernel = kernel;
	}

	/**
	 * Creates and returns a copy of this object.
	 * @see java.lang.Object#clone()
	 */
	@SuppressWarnings("unchecked")
	@Override
	public LaSVM<T> copy() throws CloneNotSupportedException {
		return (LaSVM<T>) super.clone();
	}
}
