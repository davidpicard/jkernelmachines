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
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import fr.lip6.jkernelmachines.classifier.LaSVM;
import fr.lip6.jkernelmachines.classifier.SMOSVM;
import fr.lip6.jkernelmachines.kernel.typed.DoubleGaussL2;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * Simple example comparing the training times and output of SMO and LaSVM on double vectors sampled for the normal distribution.
 * @author picard
 *
 */
public class TestLaSVM {

	public static void main(String[] args)
	{
		int dimension = 3;
		int nbPosTrain = 250;
		int nbNegTrain = 250;
		int nbPosTest = 250;
		int nbNegTest = 250;
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
		
		Collections.shuffle(train);
		
		//3. train svm
		long time = System.currentTimeMillis();
//		Kernel<double[]> k = new DoubleLinear();
		DoubleGaussL2 k = new DoubleGaussL2();
		k.setGamma(1);
		SMOSVM<double[]> svm = new SMOSVM<double[]>(k);
		svm.setC(1e3);
		svm.train(train);
		long smotime = System.currentTimeMillis() - time;
		System.out.println("SMO done");
		
		//3.1 train LaSVM
		time = System.currentTimeMillis();
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
		
		
		//7.1 compute w for smo
		double w[] = new double[dimension];
		double alpha[] = svm.getAlphas();
		for(int t = 0 ; t < train.size(); t++)
		{
			double d[] = train.get(t).sample;
			int y = train.get(t).label;
			for(int i = 0 ; i < dimension; i++)
			{
				w[i] += alpha[t] * y * d[i];
			}
		}
		System.out.println("smo: alphas : "+Arrays.toString(alpha));
		System.out.println("smo : w : "+Arrays.toString(w));
		System.out.println("smo : bias : "+svm.getB());
		System.out.println("smo : ||w|| : "+k.valueOf(w, w));
		
		//7.2 w from lasvm
		double law[] = new double[dimension];
		double laalpha[] = lasvm.getAlphas();
		System.out.println("la : alphas : "+Arrays.toString(laalpha));
		for(int t = 0 ; t < train.size(); t++)
		{
			double d[] = train.get(t).sample;
			int y = train.get(t).label;
			for(int i = 0 ; i < dimension; i++)
			{
				law[i] += alpha[t] * y * d[i];
			}
		}
		System.out.println("la : w : "+Arrays.toString(law));
		System.out.println("la : bias : "+lasvm.getB());
		System.out.println("la : ||w|| : "+k.valueOf(law, law));
		
		//8. comparing smo and peg
		double[] err = new double[laalpha.length];
		double sumerr = 0;
		for(int i = 0 ; i < err.length ; i++)
		{
			err[i] = (alpha[i]-laalpha[i])*(alpha[i]-laalpha[i]);
			sumerr += err[i];
		}
		sumerr = Math.sqrt(sumerr)/alpha.length;
		System.out.println("sumerr="+sumerr+" err : "+Arrays.toString(err));
		System.out.println("< smo, la > : "+(k.valueOf(w,law)/Math.sqrt(k.valueOf(w, w)*k.valueOf(law, law))));
		
		
		System.out.println("SMO trained in "+smotime+" ms , LaSVM trained in "+latime+" ms.");
	}
	
}
