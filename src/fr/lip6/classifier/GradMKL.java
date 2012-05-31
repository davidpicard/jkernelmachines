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
import java.util.Hashtable;
import java.util.List;

import fr.lip6.kernel.Kernel;
import fr.lip6.kernel.SimpleCacheKernel;
import fr.lip6.kernel.adaptative.ThreadedSumKernel;
import fr.lip6.type.TrainingSample;
import fr.lip6.util.DebugPrinter;

/**
 * MKL algorithm using a naive gradient descent.
 * @author picard
 *
 * @param <T>
 */
public class GradMKL<T> implements Classifier<T> {

	ArrayList<TrainingSample<T>> listOfExamples;
	ArrayList<Double> listOfExampleWeights;
	ArrayList<Kernel<T>> listOfKernels;
	ArrayList<Double> listOfKernelWeights;
	
	SMOSVM<T> svm;
	
	double stopGap = 1e-7;
	double eps_regul = 1e-3;
	double num_cleaning = 1e-7;
	double p_norm = 2;
	double C = 1e5;
	
	DebugPrinter debug = new DebugPrinter();
	
	
	public GradMKL()
	{
		listOfKernels = new ArrayList<Kernel<T>>();
		listOfKernelWeights = new ArrayList<Double>();
		listOfExamples = new ArrayList<TrainingSample<T>>();
		listOfExampleWeights = new ArrayList<Double>();
	}
	
	public void addKernel(Kernel<T> k)
	{
		listOfKernels.add(k);
		listOfKernelWeights.add(1.0);
	}
	
	@Override
	public void train(TrainingSample<T> t) {
		// TODO Auto-generated method stub
		System.err.println("Not implemented !!!");
	}

	@Override
	public void train(List<TrainingSample<T>> l) {

		long tim = System.currentTimeMillis();
		debug.println(2, "training on "+listOfKernels.size()+" kernels and "+l.size()+" examples");
		
		//1. init kernels
		ArrayList<SimpleCacheKernel<T>> kernels = new ArrayList<SimpleCacheKernel<T>>();
		ArrayList<Double> weights = new ArrayList<Double>();
		
		//normalize to cst trace and init weights to 1/N
		for(int i = 0 ; i < listOfKernels.size(); i++)
		{
			SimpleCacheKernel<T> sck = new SimpleCacheKernel<T>(listOfKernels.get(i), l);
			sck.setName(listOfKernels.get(i).toString());
			double[][] matrix = sck.getKernelMatrix(l);
			//compute trace
			double trace = 0.;
			for(int x = 0 ; x < matrix.length; x++)
			{
				trace += matrix[x][x];
			}
			//divide by trace
			for(int x = 0 ; x < matrix.length; x++)
			for(int y = x ; y < matrix.length; y++)
			{
				matrix[x][y] *= matrix.length/(double)trace;
				matrix[y][x] = matrix[x][y];
			}
			kernels.add(sck);
			weights.add(Math.pow(1/(double)listOfKernels.size(), 1/(double)p_norm));
			debug.println(3, "kernel : "+sck+" weight : "+weights.get(i));
		}
		
		
		//1 train first svm
		ThreadedSumKernel<T> tsk = new ThreadedSumKernel<T>();
		for(int i = 0 ; i < kernels.size(); i++)
			tsk.addKernel(kernels.get(i), weights.get(i));
		svm = new SMOSVM<T>(tsk);
		svm.setC(C);
		svm.train(l);
		
		//2. big loop
		double gap = 0;
		do
		{
			debug.println(3, "weights : "+weights);
			//compute sum kernel
			tsk = new ThreadedSumKernel<T>();
			for(int i = 0 ; i < kernels.size(); i++)
				tsk.addKernel(kernels.get(i), weights.get(i));
		
			//train svm
			svm.setKernel(tsk);
			svm.retrain();
			
			//compute sum of example weights and gradient direction
			double suma = computeSumAlpha();
			double [] grad = computeGradBeta(kernels, l);
			
			//perform one step
			double objEvol = performMKLStep(suma, grad, kernels, weights, l);
			
			if(objEvol < 0)
			{
				debug.println(1, "Error, performMKLStep return wrong value");
				System.exit(0);;
			}
			gap = 1 - objEvol;
			
			//compute norm
			double norm = 0;
			for(int i = 0 ; i < weights.size(); i++)
				norm += Math.pow(weights.get(i), p_norm);
			norm = Math.pow(norm, -1/(double)p_norm);
			
			debug.println(1, "objective_gap : "+gap+" norm : "+norm);
			
		}
		while(gap >= stopGap);
		
		//3. save weights
		listOfKernelWeights.clear();
		listOfKernelWeights.addAll(weights);
		
		//4. retrain svm 
		//compute sum kernel
		tsk = new ThreadedSumKernel<T>();
		for(int i = 0 ; i < kernels.size(); i++)
			tsk.addKernel(listOfKernels.get(i), listOfKernelWeights.get(i));
		//train svm
		svm.setKernel(tsk);
		svm.retrain();
		
		//5. save examples wxaeights
		listOfExamples.addAll(l);
		listOfExampleWeights.clear();
		for(double d : svm.getAlphas())
			listOfExampleWeights.add(d);

		debug.println(1, "MKL trained in "+(System.currentTimeMillis()-tim)+" milis.");
	}
	
	private double performMKLStep(double suma, double[] grad, ArrayList<SimpleCacheKernel<T>> kernels, ArrayList<Double> weights, List<TrainingSample<T>> l)
	{
		debug.print(2, ".");
		//compute objective function
		double oldObjective = +suma;
		for(int i = 0 ; i < grad.length; i++)
		{
			oldObjective -= weights.get(i)*grad[i]; 
		}
		debug.println(3, "oldObjective : "+oldObjective+" sumAlpha : "+suma);
		
		//compute optimal step
		double newBeta[] = new double[grad.length];
		
		for(int i = 0 ; i < grad.length; i++)
		{
			if(grad[i] >= 0 && weights.get(i) >= 0)
			{
				newBeta[i] = grad[i] * weights.get(i)*weights.get(i) / p_norm;
				newBeta[i] = Math.pow(newBeta[i], 1 / ((double) 1 + p_norm));
			}
			else
				newBeta[i] = 0;
		}
		
		//normalize
		double norm = 0;
		for(int i = 0 ; i < newBeta.length; i++)
			norm += Math.pow(newBeta[i], p_norm);
		norm = Math.pow(norm, -1/(double)p_norm);
		if(norm < 0)
		{
			debug.println(1, "Error normalization, norm < 0");
			return -1;
		}
		for(int i = 0 ; i < newBeta.length; i++)
			newBeta[i] *= norm;
		
		//regularize and renormalize
		double R = 0;
		for (int i = 0 ; i < kernels.size(); i++)
			R += Math.pow(weights.get(i)-newBeta[i], 2);
		R = Math.sqrt(R/(double)p_norm) * eps_regul;
		if(R < 0)
		{
			debug.println(1, "Error regularization, R < 0");
			return -1;
		}
		norm = 0;
		for(int i = 0 ; i < kernels.size(); i++)
		{
			newBeta[i] += R;
			if(newBeta[i] < num_cleaning)
				newBeta[i] = 0;
			norm += Math.pow(newBeta[i], p_norm);
		}
		norm = Math.pow(norm, -1/(double)p_norm);
		if(norm < 0)
		{
			debug.println(1, "Error normalization, norm < 0");
			return -1;
		}
		for(int i = 0 ; i < newBeta.length; i++)
			newBeta[i] *= norm;
		
		//store new weights
		for(int i = 0 ; i < weights.size(); i++)
			weights.set(i, newBeta[i]);
		
		//compute objective function
		double objective = +suma;
		for(int i = 0 ; i < grad.length; i++)
		{
			objective -= weights.get(i)*grad[i]; 
		}
		debug.println(3, "objective : "+objective+" sumAlpha : "+suma);
		
		//return objective evolution
		return objective/oldObjective;
	}
			
	
	/** calcul du gradient en chaque beta */
	private double [] computeGradBeta(ArrayList<SimpleCacheKernel<T>> kernels, List<TrainingSample<T>> l)
	{
		double grad[] = new double[kernels.size()];
		
		for(int i = 0 ; i < kernels.size(); i++)
		{
			double matrix[][] = kernels.get(i).getKernelMatrix(l);
			double a[] = svm.getAlphas();
			
			for(int x = 0 ; x < matrix.length; x++)
			{
				int l1 = l.get(x).label;
				for(int y = 0 ; y < matrix.length; y++)
				{
					int l2 = l.get(y).label;
					grad[i] += 0.5 * l1 * l2 * a[x] * a[y] * matrix[x][y];
				}
			}
		}
		
		debug.print(3, "gradDir : "+Arrays.toString(grad));
		
		return grad;
	}
	
	/** compute the sum of examples weights */
	private double computeSumAlpha()
	{
		double sum = 0;
		double[] a = svm.getAlphas();
		for(double d : a)
			sum += Math.abs(d);
		return sum;
	}

	@Override
	public double valueOf(T e) {
		
		return svm.valueOf(e);
	}
	
	public double getC() {
		return C;
	}

	public void setC(double c) {
		C = c;
	}

	public void setMKLNorm(double p)
	{
		p_norm = p;
	}
	
	public void setStopGap(double w)
	{
		stopGap = w;
	}
	
	public ArrayList<Double> getExampleWeights() {
		return listOfExampleWeights;
	}
	
	public ArrayList<Double> getKernelWeights()
	{
		return listOfKernelWeights;
	}
	
	public Hashtable<Kernel<T>, Double> getWeights()
	{
		Hashtable<Kernel<T>, Double> map = new Hashtable<Kernel<T>, Double>();
		for(int i = 0 ; i < listOfKernels.size(); i++)
			map.put(listOfKernels.get(i), listOfKernelWeights.get(i));
		return map;
	}

}
