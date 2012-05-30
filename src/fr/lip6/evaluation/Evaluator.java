package fr.lip6.evaluation;

import java.util.List;

import fr.lip6.classifier.Classifier;
import fr.lip6.type.TrainingSample;

public interface Evaluator<T> {

	/**
	 * Sets the classifier to use for evaluation
	 * @param cls the classifier
	 */
	public void setClassifier(Classifier<T> cls);
	
	/**
	 * Sets the list of training samples on which to train the classifier
	 * @param trainlist the training set
	 */
	public void setTrainingSet(List<TrainingSample<T>> trainlist);
	
	/**
	 * Sets the list of testing samples on which to evaluate the classifier
	 * @param testlist the testing set
	 */
	public void setTestingSet(List<TrainingSample<T>> testlist);
	
	/**
	 * Run the training procedure and compute score.
	 */
	public void evaluate();
	
	/**
	 * Tells the score resulting of the evaluation
	 * @return the score
	 */
	public double getScore();
	
}
