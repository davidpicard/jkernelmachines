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

    Copyright David Picard - 2012

*/
package fr.lip6.evaluation;

import java.util.List;

import fr.lip6.classifier.Classifier;
import fr.lip6.type.TrainingSample;

public class AccuracyEvaluator<T> implements Evaluator<T> {

	Classifier<T> classifier;
	List<TrainingSample<T>> trainList;
	List<TrainingSample<T>> testList;
	double accuracy;
	
	@Override
	public void setClassifier(Classifier<T> cls) {
		classifier = cls;
	}

	@Override
	public void setTrainingSet(List<TrainingSample<T>> trainlist) {
		this.trainList = trainlist;
	}

	@Override
	public void setTestingSet(List<TrainingSample<T>> testlist) {
		this.testList = testlist;
	}

	@Override
	public void evaluate() {
		if(trainList != null && testList != null) {
			long time = System.currentTimeMillis();
			classifier.train(trainList);
			System.out.println("trained in "+(System.currentTimeMillis()-time)+"ms.");
			time = System.currentTimeMillis();
			double good = 0;
			for(TrainingSample<T> t : testList) {
				double v = classifier.valueOf(t.sample);
				if(v*t.label > 0)
					good++;
			}
			accuracy = good / (double) testList.size();
			System.out.println("evaluation done in "+(System.currentTimeMillis()-time)+"ms.");
		}
	}

	@Override
	public double getScore() {
		return accuracy;
	}

}
