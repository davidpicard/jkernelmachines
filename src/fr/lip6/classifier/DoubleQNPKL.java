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
package fr.lip6.classifier;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import fr.lip6.kernel.typed.DoubleLinear;
import fr.lip6.kernel.typed.GeneralizedDoubleGaussL2;
import fr.lip6.threading.ThreadPoolServer;
import fr.lip6.threading.ThreadedMatrixOperator;
import fr.lip6.type.TrainingSample;

/**
 * <p>
 * Implementation of the QNPKL solver.<br/>
 * Original java code
 * </p>
 * 
 * <p>
 * <b>Learning geometric combinations of Gaussian kernels with alternating Quasi-Newton algorithm </b><br/>
 * Picard, D. and Thome, N. and Cord, M and Rakotomamonjy, A. <br/>
 * <i>Proceedings of the 20th ESANN conference, 2012, Bruges, 79-84</i> 
 * </p>
 * @author dpicard
 *
 * @param <T> Datatype of training samples
 */
public class DoubleQNPKL implements Classifier<double[]> {

	List<TrainingSample<double[]>> listOfExamples;
	List<Double> listOfExampleWeights;
	List<Double> listOfKernelWeights;
	int dim = 0;

	
	LaSVM<double[]> svm;
	DoubleLinear linear = new DoubleLinear();
	
	double stopGap = 1e-7;
	double num_cleaning = 1e-7;
	double p_norm = 1;
	boolean hasNorm = false;
	
	double C = 1e0;
	
	//scaling factor
	double d_lambda = 1e-1;
	
	//elementary cost matrix
	double[][] lambda_matrix = null;
		
	double oldObjective;
	
	/**
	 * Default constructor
	 */
	public DoubleQNPKL()
	{
		listOfKernelWeights = new ArrayList<Double>();
		listOfExamples = new ArrayList<TrainingSample<double[]>>();
		listOfExampleWeights = new ArrayList<Double>();
	}
	
	@Override
	public void train(TrainingSample<double[]> t) {
		if(listOfExamples == null)
			listOfExamples = new ArrayList<TrainingSample<double[]>>();
		if(!listOfExamples.contains(t))
			listOfExamples.add(t);
		train(listOfExamples);
	}

	@Override
	public void train(List<TrainingSample<double[]>> l) {

		long tim = System.currentTimeMillis();
		dim = l.get(0).sample.length;
		eprintln(2, "training on "+dim+" kernels and "+l.size()+" examples");
		
		//0. init lists
		listOfExamples = new ArrayList<TrainingSample<double[]>>();
		listOfExamples.addAll(l);
		
		//1. init kernels
		weights = new double[dim];
		
		//normalize to cst trace and init weights to 1/N
		for(int i = 0 ; i < dim; i++) {
//			weights[i] = Math.pow(1/(double)dim, 1/(double)p_norm);
			weights[i] = 1.0/dim;
//			weights[i] = 0.5/Math.sqrt(dim);
//			weights[i] = 1.0/(dim*dim);
		}
		
		
		//1 train first svm
		GeneralizedDoubleGaussL2 kernel = new GeneralizedDoubleGaussL2(weights);
		svm = trainSVM(kernel);
		double[] a = svm.getAlphas();
		//update lambda matrix before objective computation
		updateLambdaMatrix(a, kernel);
		//compute old value of objective function
		oldObjective = computeObj(a);
		eprintln(2, "+ initial objective : "+oldObjective);
		eprintln(3, "+ initial weights : "+Arrays.toString(weights));
		
		//2. big loop
		double gap = 0;
		do
		{						
			//perform one step
			double objEvol = performPKLStep();
			
			if(objEvol < 0)
			{
				eprintln(1, "Error, performPKLStep return wrong value");
				System.exit(0);;
			}
			gap = 1 - objEvol;
			
			eprintln(1, "+ objective_gap : "+(float)gap);
			eprintln(1, "+");
			
		}
		while(gap >= stopGap);
		
		
		//3. get minimal objective svm and weights
		listOfKernelWeights = new ArrayList<Double>();
		for(int i = 0 ; i < weights.length; i++)
			listOfKernelWeights.add(weights[i]);
		kernel = new GeneralizedDoubleGaussL2(weights);
		svm = trainSVM(kernel);
		//update lambdamatrix
		a = svm.getAlphas();
		updateLambdaMatrix(a, kernel);
		
		//3. compute obj
		double obj = computeObj(a);
		eprintln(2, "+ final objective : "+obj);
		
		//4. save examples weights
		listOfExamples.addAll(l);
		listOfExampleWeights.clear();
		for(double d : svm.getAlphas())
			listOfExampleWeights.add(d);

		eprintln(3, "kernel weights : "+listOfKernelWeights);
		eprintln(1, "PKL trained in "+(System.currentTimeMillis()-tim)+" milis.");
		//stop threads
		ThreadPoolServer.shutdownNow();
	}
	
	/**
	 * perform one approximate second order gradient descent step
	 * @param kernels
	 * @param weights
	 * @param l
	 * @return the objective gap
	 */
	private double performPKLStep()
	{
		
		//store new as old for the loop
		double objective = oldObjective;
		double[] oldWeights = weights;
		
		//train new svm
		GeneralizedDoubleGaussL2 k = new GeneralizedDoubleGaussL2(weights);
		LaSVM<double[]> svm = trainSVM(k);
		//update lambdamatrix
		double[] a = svm.getAlphas();
		updateLambdaMatrix(a, k);
		
		//compute grad
		double[] gNew = computeGrad(k);
		
		//estimate B
		double[] B = computeB(gNew);
		
		double lambda = 1.;
		do
		{
			//1. update weights.
			double[] wNew = new double[weights.length];
			double Z = 0;
			for(int x = 0 ; x < wNew.length ; x++) {
				wNew[x] = weights[x] - lambda * B[x] * gNew[x];
				if(wNew[x] < num_cleaning)
					wNew[x] = 0;
				if(hasNorm)
					Z += wNew[x];
			}
			
			if(hasNorm) {
				for(int x = 0 ; x < wNew.length ; x++)
					wNew[x] /= Z;
			}
				
			
			//2. retrain SVM
			k = new GeneralizedDoubleGaussL2(wNew);
			svm = trainSVM(k);
			//update lambdamatrix
			a = svm.getAlphas();
			updateLambdaMatrix(a, k);
			
			//3. compute obj
			double obj = computeObj(a);
			
			//4. check for decrease
			if(obj >= objective)
			{
				eprint(3, ".");
				lambda /= 10;
			}
			else
			{
				//store obj
				objective = obj;
				//store weights
				weights = wNew;				
			}			
			eprintln(3, "");
		}while(lambda > num_cleaning);
		
		//store gradient
		g = gNew;

		//store diff weights
		if(diffWeights == null)
		diffWeights = new double[weights.length];
		for(int x = 0 ; x < weights.length ; x++)
			diffWeights[x] = weights[x] - oldWeights[x];
		
		eprintln(3, "++++++ w : "+Arrays.toString(weights));
		
		//store new Objective
		double gap = objective / oldObjective;
		oldObjective = objective;
		
		return gap;
	}
			
	double[] diffWeights;
	double[] g;
	double[] weights;
	double[] B;
	
	/** calcul du gradient en chaque beta */
	private double[] computeGrad(GeneralizedDoubleGaussL2 kernel)
	{
		eprint(3, "++++++ g : ");
		final double grad[] = new double[dim];
		//l1 regularizer
//		Arrays.fill(grad, dim);
//		for(int x = 0 ; x < dim ; x++)
//			grad[x] *= kernel.getGammas()[x];
		
		//1 job par ligne
		int nbcpu = Runtime.getRuntime().availableProcessors();
		ThreadPoolExecutor threadPool = new ThreadPoolExecutor(nbcpu, nbcpu, 10, TimeUnit.SECONDS, new LinkedBlockingQueue<Runnable>());
		Queue<Future<?>> futures = new LinkedList<Future<?>>();
		
		class GradRunnable implements Runnable {
			GeneralizedDoubleGaussL2 kernel;
			int i;
			public GradRunnable(GeneralizedDoubleGaussL2 kernel, int i) {
				this.kernel = kernel;
				this.i = i;
			}
			
			public void run() {
				double[][] matrix = kernel.distanceMatrixUnthreaded(listOfExamples, i);
				double sum = 0;
				for(int x = 0 ; x < matrix.length ; x++)
					if(lambda_matrix[x] != null)
						for(int y = 0 ; y < matrix.length ; y++)
							sum += matrix[x][y]*lambda_matrix[x][y];
//				grad[i] = 0.5*sum/(matrix.length*matrix.length);
				grad[i] += 0.5*sum;
			}
		}
		
		for(int i = 0 ; i < grad.length ; i++) {
			Runnable r = new GradRunnable(kernel, i);
			futures.add(threadPool.submit(r));
		}
		
		//wait for all jobs
		while(!futures.isEmpty())
		{
			try {
				futures.remove().get();
			}
			catch (Exception e) {
				System.err.println("error with grad :");
				e.printStackTrace();
			}
		}

		//numerical cleaning
		for(int i = 0 ; i < grad.length; i++)
			if(Math.abs(grad[i]) < num_cleaning)
				grad[i] = 0.0;
		
		eprintln(3,Arrays.toString(grad));
		return grad;
	}
	
		
	
	/** calcul du gradient second en chaque beta */
	private double[] computeB(double[] gNew) 
	{
		if(B == null || diffWeights == null)
		{
			B = new double[gNew.length];
			Arrays.fill(B, d_lambda);
		}
		else
		{
			for(int x = 0 ; x < g.length ; x++)
			{
				double b = (gNew[x] - g[x]);
				eprint(3, " gn-g : "+b+" wn-w : "+diffWeights[x]);
				if(b != 0)
					b = diffWeights[x] / b;
				eprintln(3, " b : "+b);
				B[x] = Math.max(num_cleaning, b); // positive curvature only
				//B[x] = Math.min(b, 1.0/d_lambda); // axis cooling only
			}
		}

		eprintln(3, "++++++ B : "+Arrays.toString(B));
		return B;
	}
	
	/** compute obj */
	private double computeObj(double[] a)
	{
		double obj1 = 0, obj3 = 0;
		
		//sum of alpha
		for(double aa : a)
			obj1 += aa;
		
		

		for(int x = 0 ; x < lambda_matrix.length; x++)
		{
			if(lambda_matrix[x] == null)
				continue;
			for(int y = 0 ; y < lambda_matrix.length; y++)
			{
				if(lambda_matrix[x][y] == 0)
					continue;
				obj3 += lambda_matrix[x][y];
			}
		}
		double obj = obj1 - 0.5*obj3; 
		eprintln(3, "+++ obj : "+obj+"\t(obj1 : "+obj1+" obj3 : "+(-.5*obj3)+")");
		return obj;
	}
	
	/** compute the lambda matrix */
	private void updateLambdaMatrix(final double[] a, GeneralizedDoubleGaussL2 kernel)
	{
		final double [][] matrix = kernel.getKernelMatrix(listOfExamples);
		if(lambda_matrix == null)
			lambda_matrix = new double[matrix.length][matrix.length];
		eprintln(3, "+ update lambda matrix");
		
		ThreadedMatrixOperator factory = new ThreadedMatrixOperator()
		{
			@Override
			public void doLines(double[][] m , int from , int to) {
				for(int index = from ; index < to ; index++)
				{
					if(a[index] == 0) {
						m[index] = null;
						continue;
					}
					
					if(m[index] == null)
						m[index] = new double[matrix.length];
					
					int l1 = listOfExamples.get(index).label;
					double al1 = a[index]*l1;
					for(int j = 0 ; j < m[index].length ; j++)
					{
						int l2 = listOfExamples.get(j).label;
						m[index][j] = al1 * l2 * a[j] * matrix[index][j];
					}
				}
			}

			
		};
		
		lambda_matrix = factory.getMatrix(lambda_matrix);
	}
	
	private LaSVM<double[]> trainSVM(GeneralizedDoubleGaussL2 kernel)
	{
		LaSVM<double[]> svm = new LaSVM<double[]>(kernel);
		svm.setC(C);
		svm.setE(2);
		eprintln(3, "+ training svm");
		svm.train(listOfExamples);
		return svm;
	}
	

	@Override
	public double valueOf(double[] e) {
		
		return svm.valueOf(e);
	}
	

	private int VERBOSITY_LEVEL = 0;
	
	/**
	 * set how verbose QNPKL shall be. <br />
	 * Everything is printed to stderr. <br />
	 * none : 0 (default), few  : 1, more : 2, all : 3
	 * @param l
	 */
	public void setVerbosityLevel(int l)
	{
		VERBOSITY_LEVEL = l;
	}

	/** print() for given debug level */
	public void eprint(int level, String s)
	{
		if(VERBOSITY_LEVEL >= level)
			System.err.print(s);
	}
	
	/** println() for given debug level */
	public void eprintln(int level, String s)
	{
		if(VERBOSITY_LEVEL >= level)
			System.err.println(s);
	}

	/**
	 * Tells the hyperparameter C
	 */
	public double getC() {
		return C;
	}

	/**
	 * Sets the hyperparameter C
	 */
	public void setC(double c) {
		C = c;
	}

	/**
	 * Sets norm constraint
	 */
	public void setPNorm(double p)
	{
		p_norm = p;
	}
	
	/** Sets stopping criterion */
	public void setStopGap(double w)
	{
		stopGap = w;
	}
	
	/** Tells numerical cleaning threashold */
	public double getNum_cleaning() {
		return num_cleaning;
	}

	/** Sets numerical threshold */
	public void setNum_cleaning(double num_cleaning) {
		this.num_cleaning = num_cleaning;
	}

	/** Tells weights of training samples */
	public List<Double> getExampleWeights() {
		return listOfExampleWeights;
	}
	
	/** Tells weights of kernels */
	public List<Double> getListOfKernelWeights()
	{
		return listOfKernelWeights;
	}

	/** Tells weights of kernels as array */
	public double[] getKernelWeights()
	{
		double[] w = new double[listOfKernelWeights.size()];
		for(int x = 0 ; x < listOfKernelWeights.size(); x++)
			w[x] = listOfKernelWeights.get(x);
		return w;
	}
	
	
	class GradMAtrixOperator extends ThreadedMatrixOperator {
		
		double[] grad;
		Map<Integer, double[][]> matrices = new HashMap<Integer, double[][]>();
		
		public void addMatrix(int i, double[][] m) {
			matrices.put(i, m);
		}
		
		public void setGrad(double[] g) {
			this.grad = g;
		}
		
		public void clearMatrices() {
			matrices.clear();
		}
		
		@Override
		public void doLines(double[][] matrix , int from , int to) {

			Map<Integer, Double> summ = new HashMap<Integer, Double>();
			for(int i : matrices.keySet()) {
				double sum = 0;
				double[][] m = matrices.get(i);
				for(int x = from ; x < to; x++)
				{
					if(lambda_matrix[x] == null)
						continue;
					for(int y = 0 ; y < matrix.length; y++)
					{
						sum += m[x][y] * lambda_matrix[x][y];
					}
				}		

				summ.put(i, sum);
			}

			synchronized(grad) {
				for(int i : summ.keySet())
					grad[i] += 0.5 * summ.get(i);
			}
		}
	}

	
	/** Tells if use a norm constraint */
	public boolean isHasNorm() {
		return hasNorm;
	}

	/** Sets use of norm constraint */
	public void setHasNorm(boolean hasNorm) {
		this.hasNorm = hasNorm;
	}

}
