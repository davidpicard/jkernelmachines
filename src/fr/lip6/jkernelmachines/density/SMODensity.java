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
package fr.lip6.jkernelmachines.density;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import fr.lip6.jkernelmachines.kernel.Kernel;
import fr.lip6.jkernelmachines.util.DebugPrinter;

/**
 * Density function based on SMO algorithm.
 * 
 * @author dpicard
 *
 * @param <T> Datatype of input space
 */
public class SMODensity<T> implements DensityFunction<T>, Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 4738328902335184013L;
	
	
	private Kernel<T> K;
	private double[] alphas;
	//training set
	private ArrayList<T> set;
	
	private int size;
	
	DebugPrinter debug = new DebugPrinter();

	//parametres
	private final double epsilon=0.001;
	private double C=1;
	double tolerance = 1e-7;
	//cache d'erreur
	double cache[];
	
	/**
	 * Constructor using the specified kernel function for computing similarities among samples 
	 * @param K the kernel to use
	 */
	public SMODensity(Kernel<T> K)
	{
		this.K = K;
	}

	@Override
	public void train(T e) {
		
		if(set == null)
		{
			set = new ArrayList<T>();
		}

		set.add(e);
				
		double[] a_tmp = Arrays.copyOf(alphas, alphas.length+1);
		a_tmp[alphas.length] = 0.;
		alphas = a_tmp.clone();
		
		train();
	}

	@Override
	public void train(List<T> e) {
		if(set == null)
		{
			set = new ArrayList<T>();
		}
		
		for(T t : e)
			set.add(t);
		
		alphas = new double[set.size()];
		Arrays.fill(alphas, 0.);
		alphas[0] = 1.;
		
		size = set.size();
		
		train();
	}
	
	//calcul de l'optimisation
	private void train()
	{		
		cache = new double[size];
		Arrays.fill(cache, 1.);

		C = 1. / size;


		int nChange = 0;
		boolean bExaminerTout = true;

		int ite = 0;
		// On examine les exemples, de préférence ceux qui ne sont pas au bords (qui ne
		//  sont pas des SV).

		while (nChange > 0 || bExaminerTout)
		{
			nChange = 0;
			if (bExaminerTout)
			{
				for (int i=0;i<size;i++)
					if (examiner (i))
						nChange ++;
			}
			else
			{
				for (int i=0;i<size;i++)
					if (alphas[i] > epsilon && alphas[i] < C-epsilon)
						if (examiner (i))
							nChange ++;
			}
			if (bExaminerTout)
				bExaminerTout = false;
			else if (nChange == 0)
				bExaminerTout = true;

			ite++;
			if (ite > 1000000) 
			{
				debug.println(2, "Too many iterations...");
				break;
			}
		}
		
		debug.println(1, "trained in "+ite+" iterations.");
		
	}
	
//	 Regarde si le alpha[i1] viole la condition de KKT, et si c'est le cas,
//	 cherche un autre alpha[i2] pour l'optimisation
	private boolean examiner ( int i1)
	{ 
		// alpha[i1] doit-il est pris en compte pour optimiser ?
		//if (cache[i1]*alphas[i1] > epsilon || cache[i1]*(alphas[i1]-C) > epsilon)
		if ((cache[i1] < -tolerance && alphas[i1] < C-epsilon) || (cache[i1] > tolerance && alphas[i1] > epsilon))
		{
			// On cherche i2 de 3 fa�on diff�rentes...

			double rMax = 0;
			int i2 = alphas.length;
			for (int i=0;i<alphas.length;i++)
				if (alphas[i] > epsilon && alphas[i] < C-epsilon)
				{
					double r = Math.abs(cache[i1]-cache[i]);
					if (r > rMax)
					{
						rMax = r;
						i2 = i;
					}
				}
			if (i2 < alphas.length)
				if (optimiser (i1,i2))
				{
					return true;
				}

			int k0 = (new Random()).nextInt(alphas.length);
			for (int k=k0;k<k0+alphas.length;k++)
			{
				i2 = k % alphas.length;
				if (alphas[i2] > epsilon && alphas[i2] < C-epsilon)
					if (optimiser (i1,i2))
					{
						return true;
					}
			}
			
			// Recherche 3: Bon... et bien on va en prendre un au hazard
			k0 = (new Random()).nextInt(size);
			for (int k=k0;k<k0+size;k++)
			{
				i2 = k % size;
				if (optimiser (i2,i1))
				{
					return true;
				}
			}
			// Si on arrive ici, c'est que l'on a fait bcq de calculs pour rien
		}

		// La condition KKT est repect�e, il n'y a rien � faire
		return false;
	}

	//résolution du sous problème de manière analytique
	boolean		optimiser ( int i1, int i2)
	{
		if (i1 == i2)
			return false;


		int i;
		double delta = alphas[i1]+alphas[i2];

		double L,H;
		if (delta > C)
		{
			L = delta - C;
			H = C;
		}
		else
		{
			L = 0;
			H = delta;
		}

		if (L == H)
		{
			return false;
		}

		double k11 = K.valueOf(set.get(i1),set.get(i1));
		double k22 = K.valueOf(set.get(i2),set.get(i2));
		double k12 = K.valueOf(set.get(i1),set.get(i2));

		double a1,a2;
		double eta = 2*k12 - k11 - k22;
		if (eta < 0)
		{
			a2 = alphas[i2] + (cache[i2]-cache[i1])/eta;
			if (a2 < L) a2 = L;
			else if (a2 > H) a2 = H;
		}
		else
		{
			double c1 = eta/2;
			double c2 = cache[i1]-cache[i2] - eta * alphas[i2];
			double Lp = c1 * L * L + c2 * L;
			double Hp = c1 * H * H + c2 * H;
			if (Lp > Hp + epsilon) a2 = L;
			else if (Lp < Hp + epsilon) a2 = H;
			else a2 = alphas[i2];
		}

		if (Math.abs(a2 - alphas[i2]) < epsilon * (a2 + alphas[i2] + epsilon))
		{
			return false;
		}

		a1 = delta - a2;

		if (a1 < 0)
		{
			a2 += a1;
			a1 = 0;
		}
		else if (a1 > C)
		{
			a2 += a1-C;
			a1 = C;
		}

		double t1 = a1 - alphas[i1];
		double t2 = a2 - alphas[i2];
		for (i=0;i<alphas.length;i++)
			//if (alphas[i] > epsilon && alphas[i] < C-epsilon)
				cache[i] += t1*K.valueOf(set.get(i1),set.get(i)) + t2*K.valueOf(set.get(i2),set.get(i));

		alphas[i1] = a1;
		alphas[i2] = a2;


		return true;
	}

	
	@Override
	public double valueOf(T e) {

		double sum = 0.;
		for(int i = 0 ; i < size ; i++)
			sum += alphas[i]*K.valueOf(e, set.get(i));
		
		return sum;
	}
	
	/**
	 * Tells the weights of the training samples
	 * @return an array of double representing the weights in the training list order
	 */
	public double[] getAlphas()
	{
		return alphas;
	}

	/**
	 * Tells the hyperparameter C
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

}
