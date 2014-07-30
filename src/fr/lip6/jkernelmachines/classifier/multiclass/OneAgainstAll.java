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
package fr.lip6.jkernelmachines.classifier.multiclass;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;

import fr.lip6.jkernelmachines.classifier.Classifier;
import fr.lip6.jkernelmachines.threading.ThreadPoolServer;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.util.DebugPrinter;

/**
 * <p>
 * Multiclass classifier with N times One against All scheme.
 * </p>
 * <p>
 * The classification algorithm for each case is not set in this classifier, and
 * should be provided.
 * </p>
 * 
 * @author picard
 * 
 */
public class OneAgainstAll<T> implements MulticlassClassifier<T> {

	Classifier<T> baseClassifier;

	List<Integer> classIndices;
	List<Classifier<T>> listOfClassifiers;
	List<TrainingSample<T>> tlist;
	int nbclasses = 0;

	DebugPrinter debug = new DebugPrinter();

	/**
	 * <p>
	 * Default constructor with underlying classifier algorithm.
	 * </p>
	 * <p>
	 * The classifier given as argument is cloned N times at each training, in
	 * order to provide a binary classification for each category.
	 * </p>
	 * 
	 * @param c
	 *            the classifier class to use
	 */
	public OneAgainstAll(Classifier<T> c) {
		baseClassifier = c;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.classifier.Classifier#train(fr.lip6.type.TrainingSample)
	 */
	@Override
	public void train(TrainingSample<T> t) {
		if (tlist == null)
			tlist = new ArrayList<TrainingSample<T>>();

		tlist.add(t);
		train();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.classifier.Classifier#train(java.util.List)
	 */
	@Override
	public void train(List<TrainingSample<T>> l) {
		tlist = new ArrayList<TrainingSample<T>>();
		tlist.addAll(l);

		train();
	}

	private void train() {
		// init
		classIndices = new ArrayList<Integer>();
		listOfClassifiers = new ArrayList<Classifier<T>>();

		// count classes
		nbclasses = 0;
		for (TrainingSample<T> t : tlist) {
			if (!classIndices.contains(t.label)) {
				classIndices.add(t.label);
				nbclasses++;
				// init classifiers
				listOfClassifiers.add(null);
			}
		}
		debug.println(1, "Number of Classes: " + nbclasses);

		ThreadPoolExecutor ex = ThreadPoolServer.getThreadPoolExecutor();
		List<Future<Object>> futures = new ArrayList<>();

		// learning N one against all classifiers
		for (int id = 0; id < nbclasses; id++) {
			final int i = id;
			futures.add(ex.submit(new Callable<Object>() {

				@Override
				public Object call() throws Exception {

					Classifier<T> cls = null;
					int c = 0;

					synchronized (listOfClassifiers) {
						c = classIndices.get(i);

						// building classifier
						try {
							cls = (Classifier<T>) baseClassifier.copy();
						} catch (Exception e) {
							debug.println(1, "ERROR: Classifier not Cloneable!");
							throw new UnsupportedOperationException(
									baseClassifier.getClass().getSimpleName()
											+ " is not clonable.");
						}
					}

					debug.println(2, i + ": learning!");
					// building ad hoc trai list
					List<TrainingSample<T>> train = new ArrayList<TrainingSample<T>>();
					for (TrainingSample<T> t : tlist) {
						int y = -1;
						if (t.label == c)
							y = 1;
						train.add(new TrainingSample<T>(t.sample, y));
					}

					// training
					cls.train(train);

					// storing
					synchronized (listOfClassifiers) {
						listOfClassifiers.set(i, cls);
					}

					debug.println(1, i + ": done!");
					return null;
				}

			}));
		}

		for (Future<Object> f : futures) {
			try {
				f.get();
			} catch (InterruptedException e) {
				debug.println(1, "Error in learning on classifier");
				e.printStackTrace();
				throw new RuntimeException("Failed threading training");
			} catch (ExecutionException e) {
				debug.println(1, "Error in learning on classifier");
				e.printStackTrace();
				throw new RuntimeException("Failed threading training");
			}
		}

		ThreadPoolServer.shutdownNow(ex);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.classifier.Classifier#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(T e) {
		final T t = e;
		if (listOfClassifiers == null || listOfClassifiers.isEmpty())
			return 0;

		final double[] values = new double[listOfClassifiers.size()];

		if (nbclasses > 2 * Runtime.getRuntime().availableProcessors()) {
			ThreadPoolExecutor ex = ThreadPoolServer.getThreadPoolExecutor();
			List<Future<Object>> futures = new ArrayList<>(
					listOfClassifiers.size());
			for (int i = 0; i < listOfClassifiers.size(); i++) {
				final int id = i;
				futures.add(ex.submit(new Callable<Object>() {

					@Override
					public Object call() throws Exception {
						values[id] = listOfClassifiers.get(id).valueOf(t);
						return null;
					}

				}));
			}

			for (Future<Object> f : futures) {
				try {
					f.get();
				} catch (InterruptedException | ExecutionException e1) {
					debug.println(1, "unable to thread evaluation");
					e1.printStackTrace();
					return -1;
				}
			}

			ThreadPoolServer.shutdownNow(ex);
		}
		else {
			for(int i = 0 ; i < nbclasses ; i++) {
				values[i] = listOfClassifiers.get(i).valueOf(e);
			}
		}

		// find max output
		int imax = -1;
		double max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < listOfClassifiers.size(); i++) {
			if (values[i] > max) {
				max = values[i];
				imax = i;
			}
		}
		// return class corresponding to this output
		return classIndices.get(imax);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.classifier.multiclass.MulticlassClassifier#
	 * getConfidence(java.lang.Object)
	 */
	@Override
	public double getConfidence(T e) {
		if (listOfClassifiers == null || listOfClassifiers.isEmpty())
			return 0;

		// find max output
		double max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < listOfClassifiers.size(); i++) {
			double v = listOfClassifiers.get(i).valueOf(e);
			if (v > max) {
				max = v;
			}
		}

		return max;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.classifier.multiclass.MulticlassClassifier#
	 * getConfidences(java.lang.Object)
	 */
	@Override
	public Map<Integer, Double> getConfidences(T e) {
		if (listOfClassifiers == null || listOfClassifiers.isEmpty())
			return null;

		HashMap<Integer, Double> map = new HashMap<>();
		for (int i = 0; i < listOfClassifiers.size(); i++) {
			map.put(classIndices.get(i), listOfClassifiers.get(i).valueOf(e));
		}
		return map;
	}

	/**
	 * Returns the list of one against all classifiers used
	 * 
	 * @return the list of classifiers
	 */
	public List<Classifier<T>> getListOfClassifiers() {
		return listOfClassifiers;
	}

	/**
	 * Returns a map with class labels as keys and corresponding one against all
	 * classifiers as values
	 * 
	 * @return the map of labels, classifiers
	 */
	public Map<Integer, Classifier<T>> getMapOfClassifiers() {
		Map<Integer, Classifier<T>> map = new HashMap<Integer, Classifier<T>>();
		for (int i = 0; i < classIndices.size(); i++) {
			map.put(classIndices.get(i), listOfClassifiers.get(i));
		}
		return map;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.classifier.Classifier#copy()
	 */
	@SuppressWarnings("unchecked")
	@Override
	public OneAgainstAll<T> copy() throws CloneNotSupportedException {
		return (OneAgainstAll<T>) super.clone();
	}

}
