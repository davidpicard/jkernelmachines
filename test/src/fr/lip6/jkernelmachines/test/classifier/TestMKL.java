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
package fr.lip6.jkernelmachines.test.classifier;

import java.util.ArrayList;
import java.util.Random;

import fr.lip6.jkernelmachines.classifier.LaSVM;
import fr.lip6.jkernelmachines.classifier.SimpleMKL;
import fr.lip6.jkernelmachines.kernel.typed.DoubleGaussL2;
import fr.lip6.jkernelmachines.kernel.typed.index.IndexDoubleGaussL2;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * Simple example comparing the training times and output of SimpleMKL and LaSVM on double vectors sampled for the normal distribution.
 * @author picard
 *
 */
public class TestMKL {

	public static void main(String[] args)
	{
		int dimension = 10;
		int nbPosTrain = 25;
		int nbNegTrain = 25;
		int nbPosTest = 25;
		int nbNegTest = 25;
		double p = 1.0;
		
		Random ran = new Random(System.currentTimeMillis());
		
		ArrayList<TrainingSample<double[]>> train = new ArrayList<TrainingSample<double[]>>();
		//1. generate positive train samples
		for(int i = 0 ; i < nbPosTrain; i++)
		{
			double[] t = new double[dimension];
			for(int x = 0 ; x < dimension; x++)
			{
				t[x] = -p + ran.nextGaussian();
			}
			
			train.add(new TrainingSample<double[]>(t, 1));
		}
		//2. generate negative train samples
		for(int i = 0 ; i < nbNegTrain; i++)
		{
			double[] t = new double[dimension];
			for(int x = 0 ; x < dimension; x++)
			{
				t[x] = p + ran.nextGaussian();
			}
			
			train.add(new TrainingSample<double[]>(t, -1));
		}
		
		System.out.println("Samples generated.");
		
		
		//3. train svm
		long time = System.currentTimeMillis();
		SimpleMKL<double[]> svm = new SimpleMKL<double[]>();
		for(int i = 0 ; i < dimension ; i++) {
			IndexDoubleGaussL2 ik = new IndexDoubleGaussL2(i);
			ik.setGamma(1);
			svm.addKernel(ik);
		}
		svm.setC(1e3);
		svm.train(train);
		long smotime = System.currentTimeMillis() - time;
		System.out.println("SimpleMKL done");
		
		//3.1 train LaSVM
		time = System.currentTimeMillis();
//		Kernel<double[]> k = new DoubleLinear();
		DoubleGaussL2 k = new DoubleGaussL2();
		k.setGamma(1);
		LaSVM<double[]> lasvm = new LaSVM<double[]>(k);
		lasvm.setC(1e3);
		lasvm.train(train);
		long latime = System.currentTimeMillis() - time;
		
		
		ArrayList<TrainingSample<double[]>> test = new ArrayList<TrainingSample<double[]>>();
		//4. generate positive test samples
		for(int i = 0 ; i < nbPosTest; i++)
		{
			double[] t = new double[dimension];
			for(int x = 0 ; x < dimension; x++)
			{
				t[x] = -p + ran.nextGaussian();
			}
			
			test.add(new TrainingSample<double[]>(t, 1));
		}
		//5. generate negative test samples
		for(int i = 0 ; i < nbNegTest; i++)
		{
			double[] t = new double[dimension];
			for(int x = 0 ; x < dimension; x++)
			{
				t[x] = p + ran.nextGaussian();
			}
			
			test.add(new TrainingSample<double[]>(t, -1));
		}
		
		//6. test svm
		int nbErr = 0;
		int pegErr = 0;
		for(TrainingSample<double[]> t : test)
		{
			int y = t.label;
			double value = svm.valueOf(t.sample);
			if(y*value <= 0)
				nbErr++;
			double pegVal = lasvm.valueOf(t.sample);
			if(y*pegVal <= 0)
				pegErr++;
			
			System.out.println("y : "+y+" value : "+value+" nbErr : "+nbErr+" lasvmVal : "+pegVal+" lasvmErr : "+pegErr);
			
			
		}
		
		
		
		
		System.out.println("SimpleMKL trained in "+smotime+" ms , LaSVM trained in "+latime+" ms.");
	}
	
}
