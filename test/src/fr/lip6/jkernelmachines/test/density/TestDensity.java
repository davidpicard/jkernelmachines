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
package fr.lip6.jkernelmachines.test.density;

import java.util.ArrayList;
import java.util.Random;

import fr.lip6.jkernelmachines.density.ParzenDensity;
import fr.lip6.jkernelmachines.density.SMODensity;
import fr.lip6.jkernelmachines.kernel.typed.DoubleGaussL2;

/**
 * Test cases of density estimators using generated data.
 * @author picard
 *
 */
@Deprecated
public class TestDensity {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		int dimension = 3;
		int nbPosTrain = 250;
		int nbPosTest = 25;
		
		Random ran = new Random(System.currentTimeMillis());
		
		ArrayList<double[]> train = new ArrayList<double[]>();
		//1. generate positive train samples
		for(int i = 0 ; i < nbPosTrain; i++)
		{
			double[] t = new double[dimension];
			for(int x = 0 ; x < dimension; x++)
			{
				t[x] = ran.nextGaussian();
			}
			
			train.add(t);
		}
		System.out.println("Samples generated");
		
		
		//3. train svm
		long time = System.currentTimeMillis();
//		Kernel<double[]> k = new DoubleLinear();
		DoubleGaussL2 k = new DoubleGaussL2();
		k.setGamma(2);
		SMODensity<double[]> svm = new SMODensity<double[]>(k);
		svm.setC(100);
		svm.train(train);
		long smotime = System.currentTimeMillis() - time;
		System.out.println("SMO done in "+smotime);
		
		ParzenDensity<double[]> parzen = new ParzenDensity<double[]>(k);
		parzen.train(train);
		
		
		
		ArrayList<double[]> test = new ArrayList<double[]>();
		//4. generate positive test samples
		for(int i = 0 ; i < nbPosTest; i++)
		{
			double[] t = new double[dimension];
			for(int x = 0 ; x < dimension; x++)
			{
				t[x] = ran.nextGaussian();
			}
			
			test.add(t);
		}
		
		//6. test svm
		for(double[] t : test)
		{
			double value = svm.valueOf(t);
			double pvalue = parzen.valueOf(t);
			
			System.out.println("smo value : "+value+", parzen value : "+pvalue);
			
			
		}
		
		double[] alphas = svm.getAlphas();
		int nnz = 0;
		for(double d : alphas)
			if(d > 0)
				nnz++;
				
		System.out.println("Non zeros coefficients : "+nnz+" ("+(nnz/(double)alphas.length)+")");
		
	}

}
