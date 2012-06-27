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
package fr.lip6.classifier.multiclass;

import java.util.ArrayList;
import java.util.List;

import fr.lip6.classifier.Classifier;
import fr.lip6.type.TrainingSample;
import fr.lip6.util.DebugPrinter;

/**
 * <p>Multiclass classifier with N times One against All scheme.</p>
 * <p>
 * The classification algorithm for each case is not set in this classifier, 
 * and should be provided.
 * </p>
 * 
 * @author picard
 *
 */
public class OneAgainstAll<T> implements Classifier<T> {
	
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
	 * The classifier given as argument is cloned N times at each training, 
	 * in order to provide a binary classification for each category. 
	 * </p>
	 * @param c the classifier class to use
	 */
	public OneAgainstAll(Classifier<T> c) {
		baseClassifier = c;
	}
	

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#train(fr.lip6.type.TrainingSample)
	 */
	@Override
	public void train(TrainingSample<T> t) {
		if(tlist == null)
			tlist = new ArrayList<TrainingSample<T>>();
		
		tlist.add(t);
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#train(java.util.List)
	 */
	@Override
	public void train(List<TrainingSample<T>> l) {
		tlist = new ArrayList<TrainingSample<T>>();
		tlist.addAll(l);
		
		train();
	}
	
	
	private void train() {
		//init
		classIndices = new ArrayList<Integer>();
		listOfClassifiers = new ArrayList<Classifier<T>>();

		//count classes
		nbclasses = 0;
		for(TrainingSample<T> t : tlist) {
			if(!classIndices.contains(t.label)) {
				classIndices.add(t.label);
				nbclasses++;
				//init classifiers
				listOfClassifiers.add(null);
			}
		}
		debug.println(2, "Number of Classes: "+nbclasses);
		
		// learning N one against all classifiers
		for(int i = 0 ; i < nbclasses ; i++) {
			int c = classIndices.get(i);
			
			//building classifier
			Classifier<T> cls = null;
			try {
				 cls = (Classifier<T>) baseClassifier.copy();
			}
			catch (Exception e) {
				debug.println(1, "ERROR: Classifier not Cloneable!");
				return;
			}
			
			//building ad hoc trai list
			List<TrainingSample<T>> train = new ArrayList<TrainingSample<T>>();
			for(TrainingSample<T> t : tlist) {
				int y = -1;
				if(t.label == c)
					y = 1;
				train.add(new TrainingSample<T>(t.sample, y));
			}
			
			//training
			cls.train(train);
			
			//storing
			listOfClassifiers.set(i, cls);
		}
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(T e) {
		if(listOfClassifiers == null || listOfClassifiers.isEmpty())
			return 0;
		
		// find max output
		int imax = -1;
		double max = Double.MIN_VALUE;
		for(int i = 0 ; i < listOfClassifiers.size(); i++) {
			double v = listOfClassifiers.get(i).valueOf(e);
			if(v > max) {
				max = v;
				imax = i;
			}
		}
		//return class corresponding to this output
		return classIndices.get(imax);
	}


	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#copy()
	 */
	@SuppressWarnings("unchecked")
	@Override
	public OneAgainstAll<T> copy() throws CloneNotSupportedException {
		return (OneAgainstAll<T>)super.clone();
	}

}
