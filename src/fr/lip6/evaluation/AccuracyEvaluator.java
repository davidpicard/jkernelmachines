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
