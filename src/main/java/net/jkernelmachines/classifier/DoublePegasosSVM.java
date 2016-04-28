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
import java.util.Iterator;
import java.util.List;

import net.jkernelmachines.kernel.typed.DoubleLinear;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.type.TrainingSampleStream;
import net.jkernelmachines.util.DebugPrinter;

/**
 * <p>
 * Linear SVM classifier on arrays of double using the PEGASOS algorithm.
 * </p>
 * <p>
 * <b>Pegasos: Primal Estimated sub-GrAdient SOlver for SVM</b>
 * Shai S. Shwartz, Yoram Singer, Nathan Srebro
 * <i>In Proceedings of the 24th international conference on Machine learning (2007), pp. 807-814.</i>
 * </p>
 * @author picard
 *
 */
public class DoublePegasosSVM implements OnlineClassifier<double[]>, Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 5289136605543751554L;
	
	private DoubleLinear kernel = new DoubleLinear();
	private List<TrainingSample<double[]>> tList;
	private double[] w;
	private double b = 0;
	
	
	int T = 100000;
	int k = 10;
	double lambda = 1e-3;
	double t0 = 1.e2;
	boolean bias = true;
	
	double C = 1;
	boolean hasC = false;
	
	DebugPrinter debug = new DebugPrinter();
	private int i = 0;



	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.classifier.OnlineClassifier#train(fr.lip6.jkernelmachines.type.TrainingSample)
	 */
	@Override
	public void train(TrainingSample<double[]> t) {
		if(w == null) {
			w = new double[t.sample.length];
			b = 0;
		}
		if(tList == null) {
			tList = new ArrayList<TrainingSample<double[]>>();
		}
		tList.add(t);
		__train();
	}



	/* (non-Javadoc)
	 * @see fr.lip6.jkernelmachines.classifier.OnlineClassifier#onlineTrain(fr.lip6.jkernelmachines.type.TrainingSampleStream)
	 */
	@Override
	public void onlineTrain(TrainingSampleStream<double[]> stream) {
		TrainingSample<double[]> t = stream.nextSample();
		if(t != null) { 

			//reset w
			w = new double[t.sample.length];
			Arrays.fill(w, 0.);
			
			//reset bias
			b = 0;
			
			train(t);
			
			while((t = stream.nextSample()) != null) {
				train(t);
			}
		}
	}
	
	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#train(java.util.ArrayList)
	 */
	@Override
	public void train(List<TrainingSample<double[]>> l) {
				
		//hard limit for k
		if(k > l.size())
			k = l.size();
		
		tList = l;
		int taille = tList.get(0).sample.length; 
		//reset w
		w = new double[taille];
		Arrays.fill(w, 0.);
		
		//reset bias
		b = 0;
		
		//check if C
		if(hasC)
			lambda = 1.0 / (C * tList.size());
		
		debug.println(1, "begin training");
		long time = System.currentTimeMillis();
		List<Integer> intList = new ArrayList<Integer>();
		long tLsize = tList.size();
		for(int in = 0 ; in < tLsize ; in++)
			intList.add(in);
		for(int i = 0; i< T; i++)
		{
			__train();
			/*
			//sub sample selection
			Collections.shuffle(intList);
			ArrayList<Integer> subSampleIndices = new ArrayList<Integer>();
			subSampleIndices.addAll(intList.subList(0, k));
			
			//remove y(<w,x>-b) >= 1
			for(Iterator<Integer> iter = subSampleIndices.iterator(); iter.hasNext(); )
			{
				Integer index = iter.next();
				double[] d = tList.get(index).sample;
				int y = tList.get(index).label;
				if((kernel.valueOf(w, d) - b)*y > 1)
					iter.remove();
			}
			
			//choosing step
			double eta = 1/(double)(lambda*(i+t0));
			
			//calculate half step coefficient;
			double[] w_halfstep = new double[w.length]; 
			double omel = (1-eta*lambda);
			for(int m = 0 ; m < taille; m++)
			{
				w_halfstep[m] = omel*w[m];
			}
			double dir = 0;
			for(Iterator<Integer> iter =  subSampleIndices.iterator(); iter.hasNext(); )
			{
				Integer index = iter.next();
				TrainingSample<double[]> t = tList.get(index);
				for(int m = 0 ; m < taille; m++)
				{
					if(t.sample[m] != 0)
					{
						dir = t.label*t.sample[m];
						w_halfstep[m] += eta/(double)(k)*dir;
					}
				}

			}
			
			//b
			double b_new = 0;
			if(bias)
				for(int index : subSampleIndices)
				{
					b_new += tList.get(index).label;

				}
			
			//final step
			
			double norm = Math.sqrt(kernel.valueOf(w_halfstep, w_halfstep));
			double min = 1/Math.sqrt(lambda)/norm;
			if(min > 1)
				min = 1;
			
			double[] w_fullstep = w_halfstep.clone();
			for(int m = 0 ; m < taille; m++)
				w_fullstep[m] = w_halfstep[m] * min;
			

			
			w = w_fullstep;
			if(bias)
				b = min*( (1-eta*lambda)*b - eta/(double)k*b_new);
			else
				b = 0;

			debug.println(4, "w : "+Arrays.toString(w)+" b : "+b);
			if(T>20 && i%(T/20) == 0)
				debug.print(2, ".");
			*/
			
		}
		debug.println(2, "");
	
		
		debug.println(1, "done in "+(System.currentTimeMillis()-time)+" ms");
		debug.println(3, "w : "+Arrays.toString(w)+" b : "+b);
	}


	private void __train() {
		int taille = tList.get(0).sample.length; 
		List<Integer> intList = new ArrayList<Integer>();
		long tLsize = tList.size();
		for(int in = 0 ; in < tLsize ; in++)
			intList.add(in);
		//sub sample selection
		Collections.shuffle(intList);
		ArrayList<Integer> subSampleIndices = new ArrayList<Integer>();
		subSampleIndices.addAll(intList.subList(0, Math.min(k, intList.size())));
		
		//remove y(<w,x>-b) >= 1
		for(Iterator<Integer> iter = subSampleIndices.iterator(); iter.hasNext(); )
		{
			Integer index = iter.next();
			double[] d = tList.get(index).sample;
			int y = tList.get(index).label;
			if((kernel.valueOf(w, d) - b)*y > 1)
				iter.remove();
		}
		
		//choosing step
		double eta = 1/(double)(lambda*(i+t0));
		
		//calculate half step coefficient;
		double[] w_halfstep = new double[w.length]; 
		double omel = (1-eta*lambda);
		for(int m = 0 ; m < taille; m++)
		{
			w_halfstep[m] = omel*w[m];
		}
		double dir = 0;
		for(Iterator<Integer> iter =  subSampleIndices.iterator(); iter.hasNext(); )
		{
			Integer index = iter.next();
			TrainingSample<double[]> t = tList.get(index);
			for(int m = 0 ; m < taille; m++)
			{
				if(t.sample[m] != 0)
				{
					dir = t.label*t.sample[m];
					w_halfstep[m] += eta/(double)(k)*dir;
				}
			}

		}
		
		//b
		double b_new = 0;
		if(bias)
			for(int index : subSampleIndices)
			{
				b_new += tList.get(index).label;

			}
		
		//final step
		
		double norm = Math.sqrt(kernel.valueOf(w_halfstep, w_halfstep));
		double min = 1/Math.sqrt(lambda)/norm;
		if(min > 1)
			min = 1;
		
		double[] w_fullstep = w_halfstep.clone();
		for(int m = 0 ; m < taille; m++)
			w_fullstep[m] = w_halfstep[m] * min;
		

		
		w = w_fullstep;
		if(bias)
			b = min*( (1-eta*lambda)*b - eta/(double)k*b_new);
		else
			b = 0;

		debug.println(4, "w : "+Arrays.toString(w)+" b : "+b);
		if(T>20 && i%(T/20) == 0)
			debug.print(2, ".");
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(double[] e) {
		return kernel.valueOf(w, e)-b;
	}
	



	/**
	 * Tells the maximum number of iteration of this classifier
	 * @return the t
	 */
	public int getT() {
		return T;
	}


	/**
	 * Sets the maximum number of iterations
	 * @param t the maximum number of iteration
	 */
	public void setT(int t) {
		T = t;
	}


	/**
	 * Tells the number of samples used by this classifier to compute the subgradient
	 * @return the k
	 */
	public int getK() {
		return k;
	}


	/**
	 * Sets the number of samples on which to compute the subgradient
	 * @param k the number of samples
	 */
	public void setK(int k) {
		this.k = k;
	}


	/**
	 * Tells the learning rate of this classifier
	 * @return the lambda
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
	 * Tells the hyperplane array of this classifier
	 * @return the hyperplane coordinates
	 */
	public double[] getW() {
		return w;
	}

	/**
	 * Sets the hyperplane coordinates
	 * @param w the w to set
	 */
	public void setW(double[] w) {
		this.w = w;
	}

	/**
	 * Tells the bias b of (w*x -b)
	 * @return the bias
	 */
	public double getB() {
		return b;
	}

	/**
	 * Sets the bias term
	 * @param b the b to set
	 */
	public void setB(double b) {
		this.b = b;
	}



	/**
	 * Tells if this classifier uses a bias term
	 * @return true if a bias is used 
	 */
	public boolean isBias() {
		return bias;
	}

	/**
	 * Sets if the classifier has a bias term
	 * @param bias true for bias, false for non bias
	 */
	public void setBias(boolean bias) {
		this.bias = bias;
	}
	

	/**
	 * Tells the iteration offset
	 * @return the iteration offset
	 */
	public double getT0() {
		return t0;
	}

	/**
	 * Sets the iteration offset
	 * @param t0 the iteration offset
	 */
	public void setT0(double t0) {
		this.t0 = t0;
	}

	/**
	 * Sets C hyperparameter (automatically converted in lambda)
	 * @param c C
	 */
	public void setC(double c)
	{
		hasC = true;
		C = c;
	}
	
	/**
	 * Creates and returns a copy of this object.
	 * @see java.lang.Object#clone()
	 */
	@Override
	public DoublePegasosSVM copy() throws CloneNotSupportedException {
		return (DoublePegasosSVM) super.clone();
	}

	/**
	 * Tells the C hyperparameter, if set, else return 0
	 * @return the hyperparameter C
	 */
	public double getC() {
		if(hasC)
			return C;
		return 0.;
	}

}
