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
import java.util.Random;

import fr.lip6.jkernelmachines.classifier.DoubleSGD;
import fr.lip6.jkernelmachines.classifier.SMOSVM;
import fr.lip6.jkernelmachines.kernel.Kernel;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;
import fr.lip6.jkernelmachines.type.TrainingSample;

/**
 * Simple example comparing SMO to various primal based algorithms.
 * @author picard
 *
 */
@Deprecated
public class TestSVM {

	public static void main(String[] args)
	{
		int dimension = 3;
		int nbPosTrain = 100;
		int nbNegTrain = 100;
		int nbPosTest = 400;
		int nbNegTest = 400;
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
		
		
		//3. train svm
		Kernel<double[]> k = new DoubleLinear();
		SMOSVM<double[]> svm = new SMOSVM<double[]>(k);
		svm.setC(1e3);
		svm.train(train);
		
		//3.1 train pegasos
//		DoublePegasosSVM peg = new DoublePegasosSVM();
//		peg.setK(25);
//		peg.setT(train.size());
//		peg.setBias(false);
		DoubleSGD peg = new DoubleSGD();
//		DoubleSGDQN peg = new DoubleSGDQN();
//		peg.setC(1e3);
		peg.train(train);
		
		
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
			if(y*value < 0)
				nbErr++;
			double pegVal = peg.valueOf(t.sample);
			if(y*pegVal < 0)
				pegErr++;
			
			System.out.println("y : "+y+" value : "+value+" nbErr : "+nbErr+" pegVal : "+pegVal+" pegErr : "+pegErr);
			
			
		}
		
		//7. alphas from svm
		System.out.println("smo : alphas : "+Arrays.toString(svm.getAlphas()));
		
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
		System.out.println("smo : w : "+Arrays.toString(w));
		System.out.println("smo : bias : "+svm.getB());
		System.out.println("smo : ||w|| : "+k.valueOf(w, w));
		
		//7.2 w from pegasos
		System.out.println("peg : w : "+Arrays.toString(peg.getW()));
//		System.out.println("peg : bias : "+peg.getB());
		System.out.println("peg : ||w|| : "+k.valueOf(peg.getW(), peg.getW()));
		
		//8. comparing smo and peg
		System.out.println("< smo, peg > : "+(k.valueOf(w, peg.getW())/Math.sqrt(k.valueOf(w, w)*k.valueOf(peg.getW(), peg.getW()))));
		
	}
	
}
