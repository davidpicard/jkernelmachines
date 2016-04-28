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
package net.jkernelmachines.example;

import java.util.List;

import net.jkernelmachines.classifier.BudgetSDCA;
import net.jkernelmachines.classifier.Classifier;
import net.jkernelmachines.classifier.LaSVM;
import net.jkernelmachines.classifier.LaSVMI;
import net.jkernelmachines.classifier.NystromLSSVM;
import net.jkernelmachines.classifier.SDCA;
import net.jkernelmachines.classifier.SMOSVM;
import net.jkernelmachines.evaluation.AccuracyEvaluator;
import net.jkernelmachines.evaluation.RandomSplitCrossValidation;
import net.jkernelmachines.io.ArffImporter;
import net.jkernelmachines.io.LibSVMImporter;
import net.jkernelmachines.kernel.Kernel;
import net.jkernelmachines.kernel.typed.DoubleGaussL2;
import net.jkernelmachines.kernel.typed.DoubleLinear;
import net.jkernelmachines.projection.DoublePCA;
import net.jkernelmachines.type.TrainingSample;
import net.jkernelmachines.util.DebugPrinter;

/**
 * <p>
 * This is a more complex example of how to code with JKernelMachines than can
 * be used as a stand-alone program.
 * </p>
 * <p>
 * It reads data from an input file in the libsvm format, and performs a cross-validation evaluation.
 * Optional parameters include the number of test to perform, the percentage of data to keep from training, the type of kernel, and the svm algorithm.
 * Launch without argument to get the following help:
 * </p>
 * CrossValidationExample -f file [-p percent] [-n nbtests] [-k kernel] [-a algorithm] [-pca type] [-vvv]
 * <ul>
 *	<li>-f file: the data file in libsvm format</li>
 *	<li>-p percent: the percent of data to keep for training</li>
 *	<li>-n nbtests: the number of test to perform during crossvalidation</li>
 *	<li>-k kernel: the type of kernel (linear or gauss, default gauss)</li>
 *	<li>-a algorithm: type of SVM algorithm(lasvm, lasvmi, smo, nlssvm default lasvm)</li>
 *  <li>-pca type: perform a PCA as preprocessing (no, yes, white, default no)</li>
 *	<li>-v: verbose (v few, vv lot, vvv insane, default none)</li>
 * </ul>
 * 
 * @author David Picard
 * 
 */
public class CrossValidationExample {

	/**
	 * @param args main arguments
	 */
	public static void main(String[] args) {

		String file = "";
		double percent = 0.8;
		int nbtest = 10;
		Kernel<double[]> kernel = null;
		Classifier<double[]> svm = null;
		int hasPCA = 0;

		// parsing options
		try {
			for (int i = 0; i < args.length; i++) {
				// parsing file
				if (args[i].equalsIgnoreCase("-f")) {
					i++;
					file = args[i];
				}
				// split percent
				else if (args[i].equalsIgnoreCase("-p")) {
					i++;
					percent = Double.parseDouble(args[i]);
				}
				// number of tests
				else if (args[i].equalsIgnoreCase("-n")) {
					i++;
					nbtest = Integer.parseInt(args[i]);
				}
				// kernel type
				else if (args[i].equalsIgnoreCase("-k")) {
					i++;

					if (args[i].equalsIgnoreCase("gauss")) {
						kernel = new DoubleGaussL2();
					} else {
						kernel = new DoubleLinear();
					}
				}
				// algorithm
				else if (args[i].equalsIgnoreCase("-a")) {
					i++;
					if (args[i].equalsIgnoreCase("lasvmi")) {
						svm = new LaSVMI<double[]>(kernel);
					} else if (args[i].equalsIgnoreCase("sdca")) {
						svm = new SDCA<double[]>(kernel);
					} else if (args[i].equalsIgnoreCase("bsdca")) {
						BudgetSDCA<double[]> bsdca = new BudgetSDCA<double[]>(kernel);
						bsdca.setBudget(128);
						svm = bsdca;
					}  else if (args[i].equalsIgnoreCase("smo")) {
						svm = new SMOSVM<double[]>(kernel);
					} else if (args[i].equalsIgnoreCase("nlssvm")) {
						NystromLSSVM<double[]> nlssvm = new NystromLSSVM<double[]>(kernel);
						nlssvm.setPercent(0.1);
						svm = nlssvm;
						
					}  else { // default lasvm
						svm = new LaSVM<double[]>(kernel);
					}
				}
				else if (args[i].equalsIgnoreCase("-pca")) {
					i++;
					if(args[i].equalsIgnoreCase("yes")) {
						hasPCA = 1;
					}
					else if(args[i].equalsIgnoreCase("white")) {
						hasPCA = 2;
					}
				}
				// verbose
				else if (args[i].equalsIgnoreCase("-v")) {
					DebugPrinter.DEBUG_LEVEL = 2;
				} else if (args[i].equalsIgnoreCase("-vv")) {
					DebugPrinter.DEBUG_LEVEL = 3;
				} else if (args[i].equalsIgnoreCase("-vvv")) {
					DebugPrinter.DEBUG_LEVEL = 4;
				}
			}

		} catch (Exception e) {
			printHelp();
			System.exit(-1);
		}

		// check option
		if (file.equalsIgnoreCase("") || kernel == null || svm == null) {
			printHelp();
			System.exit(-1);
		}

		// read data
		List<TrainingSample<double[]>> list = null;
		try {
			if(file.endsWith(".arff")) {
				list = ArffImporter.importFromFile(file);
			}
			else {
				list = LibSVMImporter.importFromFile(file);
			}
		} catch (Exception e) {
			System.out.println("Wrong data file");
			printHelp();
			System.exit(-1);
		}
		if (list == null) {
			System.out.println("Wrong data file");
			printHelp();
			System.exit(-1);
		}
		// perform PCA
		if(hasPCA == 1) {
			DoublePCA pca = new DoublePCA();
			pca.train(list);
			list = pca.projectList(list);
		}
		else if(hasPCA == 2) {
			DoublePCA pca = new DoublePCA();
			pca.train(list);
			list = pca.projectList(list, true);
		}

		// initialize CV
		AccuracyEvaluator<double[]> ev = new AccuracyEvaluator<double[]>();
		RandomSplitCrossValidation<double[]> cv = new RandomSplitCrossValidation<double[]>(
				svm, list, ev);
		cv.setNbTest(nbtest);
		cv.setTrainPercent(percent);

		// do cv
		long tim = System.currentTimeMillis();
		cv.run();
		tim = System.currentTimeMillis() - tim;

		// print result
		System.out.println("Accuracy: " + cv.getAverageScore() + " +/- "
				+ cv.getStdDevScore()+" in "+tim/nbtest+"ms");
	}

	private static void printHelp() {
		System.out
				.println("CrossValidationExample -f file [-p percent] [-n nbtests] [-k kernel] [-a algorithm] [-vvv]");
		System.out.println("\t-f file: the data file in libsvm format");
		System.out
				.println("\t-p percent: the percent of data to keep for training");
		System.out
				.println("\t-n nbtests: the number of test to perform during crossvalidation");
		System.out
				.println("\t-k kernel: the type of kernel (linear or gauss, default gauss)");
		System.out
				.println("\t-a algorithm: type of SVM algorithm(lasvm, lasvmi, smo, default lasvm)");
		System.out
				.println("\t-pca type: perform a PCA as preprocessing(no, yes, white, default no");
		System.out
				.println("\t-v: verbose (v few, vv lot, vvv insane, default none)");
	}

}
