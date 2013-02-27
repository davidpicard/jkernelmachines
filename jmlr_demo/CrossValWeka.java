import java.io.File;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.LibSVMLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;

/**
 * 
 */

/**
 * @author picard
 *
 */
public class CrossValWeka {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws Exception {
		
		LibSVMLoader libsvm = new LibSVMLoader();
		libsvm.setFile(new File(args[0]));
		
		//read from libsvm format files
		Instances instances = libsvm.getDataSet();
		
		//convert to nominal class argument to avoid 
		// exception in weka's smo :-/
		NumericToNominal filter = new NumericToNominal();
		filter.setInputFormat(instances);
		instances = Filter.useFilter(instances, filter);
		
		
		Random ran = new Random(System.currentTimeMillis());
		instances.randomize(ran);
		
		int max = 20;
		double[] acc = new double[max];
		while(max > 0) {
			
			//copy and randomize instances
			Instances full = new Instances(instances);
			full.randomize(ran);
			
			// using 5 fold CV to emulate the 80-20 random split of jkms
			Instances train = full.trainCV(5, 0);
			Instances test = full.testCV(5, 0);
			
			// new svm with rbf kernel
			SMO smo = new SMO();
			String[] options = { "-L 1e-15", "-P 1e-15" , "-N 2"};
			smo.setOptions(options);
			smo.setC(1.0); //same as default value for jkms
			RBFKernel rbf = new RBFKernel();
			rbf.setGamma(0.1); //same as default value for jkms
			smo.setKernel(rbf);
			
			smo.buildClassifier(train);
			
			
			Evaluation eval = new Evaluation(train);
			eval.evaluateModel(smo, test);
			
//			System.out.println(eval.toSummaryString("results:\n", false));
			acc[acc.length - max] = eval.pctCorrect();
//			System.out.println("accuracy: "+eval.pctCorrect());
			
			max--;			
		}
		
		double mu = 0;
		for(double d : acc) {
			mu += d;
		}
		mu /= acc.length;
		
		double std = 0;
		for(double d : acc) {
			std += (d-mu)*(d-mu);
		}
		std = Math.sqrt(std/acc.length);
		
		// print comparable score :-)
		System.out.println("mean accuracy : "+mu+" +/- "+std);
		

	}

}
