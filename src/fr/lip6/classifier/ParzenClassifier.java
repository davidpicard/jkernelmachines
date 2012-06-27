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
package fr.lip6.classifier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import fr.lip6.kernel.Kernel;
import fr.lip6.type.TrainingSample;

/**
 * Classification tool using a Parzen window
 * @author dpicard
 *
 * @param <T> type of input space
 */
public class ParzenClassifier<T> implements Classifier<T>, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 5147554432765939157L;
	
	
	Kernel<T> kernel;
	ArrayList<TrainingSample<T>> ts;
	
	public ParzenClassifier(Kernel<T> kernel)
	{
		this.kernel = kernel;
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#train(java.lang.Object, int)
	 */
	public void train(TrainingSample<T> t) {

		if(ts == null)
		{
			ts = new ArrayList<TrainingSample<T>>();
			
		}
		ts.add(t);

		
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#train(T[], int[])
	 */
	public void train(List<TrainingSample<T>> t) {

		ts = new ArrayList<TrainingSample<T>>(t);
	}

	/* (non-Javadoc)
	 * @see fr.lip6.classifier.Classifier#valueOf(java.lang.Object)
	 */
	public double valueOf(T e) {

		double sum = 0.;
		for(int i = 0 ; i < ts.size(); i++)
		{
			TrainingSample<T> t = ts.get(i);
			sum += t.label * kernel.valueOf(t.sample, e);
		}
		
		sum /= ts.size();
		
		return sum;
	}
	

	/**
	 * Creates and returns a copy of this object.
	 * @see java.lang.Object#clone()
	 */
	@SuppressWarnings("unchecked")
	@Override
	public ParzenClassifier<T> clone() throws CloneNotSupportedException {
		return (ParzenClassifier<T>) super.clone();
	}
	
}
