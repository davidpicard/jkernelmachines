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

import java.util.Arrays;
import java.util.List;
import java.util.Vector;

import fr.lip6.kernel.Kernel;
import fr.lip6.kernel.typed.DoubleLinear;
import fr.lip6.type.TrainingSample;

/**
 * <p>
 * Linear SVM classifier using SGDQN algorithm.
 * </p>
 * <p>
 * <b>Careful Quasi-Newton Stochastic Gradient Descent</b><br />
 * Antoine Bordes, Léon Bottou, and Patrick Gallinari<br />
 * <i>J. Mach. Learn. Res. 10 (December 2009), 1737-1754.</i>
 * </p>
 * 
 * @author picard
 *
 */
public class DoubleSGDQN implements Classifier<double[]> {
	
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
	
	//used loss function
	private int loss = HINGELOSS;
	
	//svm hyperplane
	private double[] w = null;
	private double[] Bc = null;
	//skipping decay update parameter
	private int skip = 0;
	private int count;
	private long t;
	private long t0 = 0;
	private double lambda = 1e-4;
	private int epochs = 5;
	
	double[] mean;
	double var[];
	double eps = 1e-2;
	
	boolean hasC = false;
	double C = 1.0;
	
	/**
	 * Sets the verbosity of training procedure (if true, details are printed during learning)
	 */
	public static boolean VERBOSE = false;
	
	//pretraitement
	private boolean normalize = false;
	
	//le noyau linéaire
	private final Kernel<double[]> dot = new DoubleLinear();
	
	List<TrainingSample<double[]>> tlist;
	

	@Override
	public void train(TrainingSample<double[]> t) {
		if(tlist == null)
			tlist = new Vector<TrainingSample<double[]>>();
		
		tlist.add(t);
		
		train(tlist);
		
	}

	@Override
	public void train(List<TrainingSample<double[]>> l) {
		
		//0. copy samples
		tlist = new Vector<TrainingSample<double[]>>();
		tlist.addAll(l);
		int size = tlist.size();
		//check if C
		
		if(hasC)
			lambda = 1.0 / (C * tlist.size());
		
		if(normalize)
		{
			//compute mean & variance
			mean = new double[tlist.get(0).sample.length];
			var = new double[mean.length];
			
			for(TrainingSample<double[]> x : tlist)
				for(int i = 0 ; i < x.sample.length ; i++)
					mean[i] += x.sample[i];
			//scale
			for(int i = 0 ; i < mean.length ; i++)
				mean[i] /= size;
			if(VERBOSE)
				System.out.println("||mean|| = "+dot.valueOf(mean, mean));
			//var
			for(TrainingSample<double[]> x : tlist)
				for(int i = 0 ; i < x.sample.length ; i++)
				{
					double s = (x.sample[i]-mean[i]); 
					var[i] += s*s;
				}
			for(int i = 0 ; i < var.length ; i++)
			{
				var[i] = var[i]/size;
				
				if(var[i] == 0.)
					var[i] = 0.;
				else
					var[i] = Math.sqrt(1./var[i]);
			}
			if(VERBOSE)
				System.out.println("||var|| = "+dot.valueOf(var, var));
			
			//center reduce
			for(TrainingSample<double[]> x : tlist)
			{
				for(int i = 0 ; i < x.sample.length ; i++)
				{
					x.sample[i] = (x.sample[i] - mean[i])*var[i];
				}
			}
		}
		
		long time = System.currentTimeMillis();
		//1. init
		initSVM();
		if(VERBOSE)
			System.out.println("dimension of w : "+w.length);
		
		//2. t0
		t0 = determineT0(0 , tlist.size()/10);
		if(VERBOSE)
			System.out.println("t0 set to "+t0+" ("+(System.currentTimeMillis()-time)+" ms.)");
		
		//3. train the number of epoch
		initSVM();
		calibrate(0, size - 1);
		long tmptime = System.currentTimeMillis();
		for(int i = 0 ; i < epochs ; i++)
		{
			train(0, tlist.size()-1);
			if(VERBOSE)
			{
				long t = System.currentTimeMillis();
				System.out.println("epoch "+i+" time : "+(t - tmptime)+" ms.");
				tmptime = t;
			}
		}
		if(VERBOSE)
			System.out.println("done in "+(System.currentTimeMillis()-time)+" ms.");
		
	}

	@Override
	public double valueOf(double[] e) {
		if(normalize)
		{
			double[] x = e.clone();
			for(int i = 0 ; i < x.length ; i++)
			{
				x[i] = (x[i] - mean[i])*var[i];
			}
			return dot.valueOf(w, x);
		}
		return dot.valueOf(w, e);
	}
	
	private void initSVM()
	{
		w = new double[tlist.get(0).sample.length];
		Bc = new double[w.length];
		Arrays.fill(Bc, 1./lambda);
		t = t0;
	}
	
	private void train(int from, int to)
	{
		count = skip;
		boolean updateB = false;
		
		for(int i = from ; i <= to ; i++)
		{
			TrainingSample<double[]> tx = tlist.get(i);
			double[] x = tx.sample;
			double y = tx.label;
			double z = y * dot.valueOf(w, x);
			
			double eta = 1.0 / t;
			
			if(updateB == true)
			{
				if( (loss < LOGLOSS && z < 1) || loss >= LOGLOSS)
				{
					double[] w_1 = w.clone();
					double loss_1 = dloss(z);
					
					for(int d = 0 ; d < w.length ; d++)
						w[d] += x[d] * Bc[d] * eta * loss_1 * y;
					
					double z2 = y * dot.valueOf(w,x);
				    double diffloss = dloss(z2) - loss_1;
				    
				    if(diffloss != 0)
				    {
				    	double B[] = computeRatio(x, lambda, w_1, w, y * diffloss);
				    	
				    	if(t > skip)
				    	{
				    		combineAndClip(Bc, (t-skip)/(double)(t+skip) , B , 2.*skip/(double)(t+skip), 1/(100.*lambda), 100./lambda);
				    	}
				    	else
				    	{

						    combineAndClip(Bc, t/(double)(t+skip),B,skip/(double)(t+skip),1/(100.*lambda),100./lambda);
				    	}
				    	
				    }
					
				}	
				updateB = false;
			}
			else
			{
				if(--count <= 0)
				{

					for(int d = 0 ; d < w.length ; d++)
						w[d] += -skip*lambda*eta * Bc[d];

					count = skip;
					updateB=true;
				}      
				if( (loss <  LOGLOSS && z < 1) || loss >= LOGLOSS)
				{
					for(int d = 0 ; d < w.length ; d++)
						w[d] += x[d] * eta * dloss(z)*y * Bc[d];
				}
			}
			
			t += 1;
		}
	}
	
	/** test the objective function on a subsample of training set */
	private double test(int from, int to) {
		double cost = 0;
		for (int i = from; i <= to; i++) {
			TrainingSample<double[]> tx = tlist.get(i);
			double[] x = tx.sample;
			double y = tx.label;
			double z = y * dot.valueOf(w, x);
			if ((loss < LOGLOSS && z < 1) || loss >= LOGLOSS)
				cost += loss(z);
		}
		int n = to - from + 1;
		double loss = cost / n;
		cost = loss + 0.5 * lambda * dot.valueOf(w,w);
		  
		  return cost;
	}

	private void calibrate(int from, int to)
	{
		//estimation de la parcimonie des données
		double n = 0;
		double r = 0;

		for(int j = from ; j <= to ; j++,n++)
		{
			double[] x = tlist.get(j).sample;
			n += 1;
			for(double d : x)
				if(d != 0)
					r++;
		}
		skip = (int) ((8 * n * w.length) / r);
	}
	
	
	private double[] computeRatio(double[] x , double lambda , double[] w_1 , double[] w , double loss)
	{
		double[] r = new double[x.length];
		
		for(int d = 0 ; d < x.length ; d++)
		{
			double diffw = w_1[d]-w[d];
			if(diffw != 0)
				r[d] = diffw/ (lambda*diffw+ loss*x[d]);
			else
				r[d] = 1/lambda;	
		}
		
		return r;
	}

	
	private void combineAndClip(double[] bc, double c1, double[] b, double c2,
			double min, double max) {
		
		for(int x = 0 ; x < bc.length ; x++)
		{
			bc[x] = bc[x] * c1 + b[x] * c2;
			bc[x] = Math.min(Math.max(bc[x], min), max);
		}
		
	}
	
	
	//determiner le t0 optimal sur un subset du training set
	private long determineT0(int from, int to)
	{  
		long t0 = 1;
		long t0tmp = 1;
		double lowest_cost=Double.MAX_VALUE;
		for (int i=0; i<=10; i++)
		{
			initSVM();
			this.t0 = t0tmp;
			calibrate(from, to);

//			for (int ep=0; ep<epochs; ep++)
				train(from, to);


			double cost = test(from, to);
			if (cost<lowest_cost && !Double.isNaN(cost)) // check for NaN
			{
				t0 = t0tmp;
				lowest_cost=cost;
			}      
			t0tmp=t0tmp*10;
		}	 
		return t0;
	}

	
	/**
	 * fonction de coût utilisée
	 * @param z
	 * @return
	 */
	private double loss(double z)
	{
		switch(loss)
		{
		case LOGLOSS : 
			if (z >= 0)
				return Math.log(1+Math.exp(-z));
			else
				return -z + Math.log(1+Math.exp(z));
		case LOGLOSSMARGIN :
			if (z >= 1)
				return Math.log(1+Math.exp(1-z));
			else
				return 1-z + Math.log(1+Math.exp(z-1));
		case SMOOTHHINGELOSS : 
			if (z < 0)
				return 0.5 - z;
			if (z < 1)
				return 0.5 * (1-z) * (1-z);
			return 0;
		case SQUAREDHINGELOSS :
			if (z < 1)
				return 0.5 * (1 - z) * (1 - z);
			return 0;
		case HINGELOSS :
			if (z < 1)
				return 1 - z;
			return 0;
		}
		return 0;
	}
	

	 
	private double dloss(double z)
	{
		switch(loss)
		{
		case LOGLOSS :
			if (z < 0)
				return 1 / (Math.exp(z) + 1);
			double ez = Math.exp(-z);
			return ez / (ez + 1);
		case LOGLOSSMARGIN :
			if (z < 1)
				return 1 / (Math.exp(z-1) + 1);
			ez = Math.exp(1-z);
			return ez / (ez + 1);
		case SMOOTHHINGELOSS :
			if (z < 0)
				return 1;
			if (z < 1)
				return 1-z;
			return 0;
		case SQUAREDHINGELOSS :
			if (z < 1)
				return (1 - z);
			return 0;
		default :
			if (z < 1)
				return 1;
			return 0;
		}
	}
	

	/**
	 * Tells the type of loss used  by this classifier
	 * @return an integer representing the loss type 
	 */
	public int getLoss() {
		return loss;
	}

	/**
	 * Sets the type of loos used by this classifier
	 * @param loss an integer value representing the loss (default: HINGELOSS)
	 */
	public void setLoss(int loss) {
		this.loss = loss;
	}

	/**
	 * Tells the array of coordinates of the hyperplane used by this classifier
	 * @return the array of coordinates
	 */
	public double[] getW() {
		return w;
	}

	/**
	 * Sets the array of coordinate used by this classifier
	 * @param w the array of coordinates
	 */
	public void setW(double[] w) {
		this.w = w;
	}

	/**
	 * Tells the learning rate lambda
	 * @return the learning rate lambda
	 */
	public double getLambda() {
		return lambda;
	}

	/**
	 * Sets the learning rate lambda
	 * @param lambda the learning rate
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	/**
	 * Tells the number of epochs used for training this classifier
	 * @return the number of epochs
	 */
	public int getEpochs() {
		return epochs;
	}

	/**
	 * Sets the number of epochs used for training this classifier
	 * @param epochs the number of epochs
	 */
	public void setEpochs(int epochs) {
		this.epochs = epochs;
	}

	/**
	 * Tells if training datas are centered/reduced as preprocessing before learning
	 * @return true if center/reduce
	 */
	public boolean isNormalize() {
		return normalize;
	}

	/**
	 * Sets if training datas are centered/reduced as preprocessing before learning
	 * @param normalize true for center/reduce (default false)
	 */
	public void setNormalize(boolean normalize) {
		this.normalize = normalize;
	}

	/**
	 * Tells the C hyperparameter
	 * @return C
	 */
	public double getC() {
		return C;
	}

	/**
	 * Set the C hyperparameter (automatically converted to learning rate lambda)
	 * @param c the hyperparameter C
	 */
	public void setC(double c) {
		C = c;
		hasC = true;
	}

	/**
	 * Creates and returns a copy of this object.
	 * @see java.lang.Object#clone()
	 */
	@Override
	public DoubleSGDQN copy() throws CloneNotSupportedException {
		return (DoubleSGDQN) super.clone();
	}
}
