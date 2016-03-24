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

    Copyright David Picard - 2013

 */
package fr.lip6.jkernelmachines.classifier;

import static java.lang.Math.abs;
import static java.lang.Math.max;
import static java.lang.Math.min;

import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import fr.lip6.jkernelmachines.classifier.Classifier;
import fr.lip6.jkernelmachines.classifier.KernelSVM;
import fr.lip6.jkernelmachines.classifier.OnlineClassifier;
import fr.lip6.jkernelmachines.kernel.Kernel;
import fr.lip6.jkernelmachines.type.ListSampleStream;
import fr.lip6.jkernelmachines.type.TrainingSample;
import fr.lip6.jkernelmachines.type.TrainingSampleStream;
import fr.lip6.jkernelmachines.util.DebugPrinter;

/**
 * <p>
 * SDCA svm algorithm from Shalev-Shwartz adapted for budget constraints.
 * </p>
 * <p>
 * Stochastic Dual Coordinate Ascent Methods for Regularized Loss Minimization,
 * <br/>
 * Shai Shalev-Shwartz, Tong Zhang<br/>
 * JMLR, 2013.
 * </p>
 * 
 * @author picard
 * 
 */
public class BudgetSDCA<T> implements KernelSVM<T>, OnlineClassifier<T> {

	private class SV {
		TrainingSample<T> sample;
		double alpha;
		double z;

		public SV(TrainingSample<T> x, double a, double v) {
			sample = x;
			alpha = a;
			z = v;
		}

		@Override
		public int hashCode() {
			return sample.hashCode();
		}

		@Override
		public boolean equals(Object obj) {
			return sample.hashCode() == obj.hashCode();
		}
	}

	Kernel<T> kernel;
	List<SV> train;

	double C = 1.0;
	int E = 2;

	int budget = 256;
	double capacity = 1.05;
	double eps = 1e-10;

	DebugPrinter debug = new DebugPrinter();

	/**
	 * @param kernel
	 */
	public BudgetSDCA(Kernel<T> kernel) {
		super();
		this.kernel = kernel;
		train = new LinkedList<>();
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.classifier.Classifier#train(fr.lip6.jkernelmachines
	 * .type.TrainingSample)
	 */
	@Override
	public void train(TrainingSample<T> t) {
		addSample(t);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#train(java.util.List)
	 */
	@Override
	public void train(List<TrainingSample<T>> l) {

		ListSampleStream<T> strain = new ListSampleStream<>(l);
		onlineTrain(strain);
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.classifier.OnlineClassifier#onlineTrain(fr.lip6
	 * .jkernelmachines.type.TrainingSampleStream)
	 */
	@Override
	public void onlineTrain(TrainingSampleStream<T> stream) {

		TrainingSample<T> t;
		int i = 0;
		while ((t = stream.nextSample()) != null) {
			train(t);
			i++;
			debug.print(1, "  " + i + " (" + train.size() + "/" + budget + ")");
		}
		if (train.size() > budget) {
			prune();
		}
		reprocess();
		debug.print(
				1,
				"\r                                                                                \r");

	}

	public boolean prune() {
		if (train.size() > budget) {
			reprocess();
			pruneZeroAlpha();

			// prune highest error
			if (train.size() > budget) {

				pruneError();

				// prune lowest sum kernel
				if (train.size() > budget) {
					reprocess();
					pruneLowAlpha();
				}
				return true;
			}
		}
		return false;
	}

	// reprocess all SV from highest error
	private final void reprocess() {
		long t = System.currentTimeMillis();
		for (int e = 0; e < E; e++) {
			synchronized (train) {
				Collections.sort(train, new Comparator<SV>() {

					@Override
					public int compare(BudgetSDCA<T>.SV o1, BudgetSDCA<T>.SV o2) {
						return Double.compare(o1.z * o1.sample.label, o2.z
								* o2.sample.label);
					}
				});

				for (int i = 0; i < train.size(); i++) {
					updateNoCache(i);
				}
			}
		}
		debug.print(2, "  rp: " + (System.currentTimeMillis() - t) + "ms");
	}

	private final void pruneZeroAlpha() {

		long t = System.currentTimeMillis();
		int rem = 0;
		synchronized (train) {
			for (Iterator<SV> ite = train.iterator(); ite.hasNext();) {
				if (ite.next().alpha == 0) {
					ite.remove();
					rem++;
				}
			}
		}
		debug.print(2, "  pz: rem: " + rem + " "
				+ (System.currentTimeMillis() - t) + "ms");
	}

	// prune highest error
	private final void pruneError() {
		synchronized (train) {
			long t = System.currentTimeMillis();
			Collections.sort(train, new Comparator<SV>() {

				@Override
				public int compare(BudgetSDCA<T>.SV o1, BudgetSDCA<T>.SV o2) {
					return Double.compare(o1.z * o1.sample.label, o2.z
							* o2.sample.label);
				}
			});
			SV thsv = train.get(train.size() - budget);
			double th = thsv.z * thsv.sample.label;
			if (th >= 1)
				th = 1;
			debug.print(2, " th=" + th);
			List<SV> removed = new LinkedList<>();
			while (true) {
				SV sv = train.get(0);
				if (sv.z * sv.sample.label < th) {
					removed.add(train.remove(0));
				} else {
					break;
				}
			}
			for (SV sv : removed) {
				if (sv.alpha != 0) {
					for (SV s : train) {
						s.z -= sv.alpha
								* kernel.valueOf(sv.sample.sample,
										s.sample.sample);
					}
				}
			}
			debug.print(2, "  pe: rem=" + removed.size());
			debug.print(2, " " + (System.currentTimeMillis() - t) + "ms");
		}
	}

	// prune lowest abs(alpha_i)
	private final void pruneLowAlpha() {
		synchronized (train) {
			long t = System.currentTimeMillis();
			List<SV> removed = new LinkedList<>();
			if (train.size() > budget) {
				Collections.sort(train, new Comparator<SV>() {

					@Override
					public int compare(BudgetSDCA<T>.SV o1, BudgetSDCA<T>.SV o2) {
						return Double.compare(abs(o1.alpha), abs(o2.alpha));
					}

				});
				while (train.size() > budget) {
					removed.add(train.remove(0));
				}
			}
			for (SV sv : removed) {
				if (sv.alpha != 0) {
					for (SV s : train) {
						double k = kernel.valueOf(sv.sample.sample,
								s.sample.sample);
						s.z -= sv.alpha * k;
					}
				}
			}
			debug.print(
					2,
					"  pl: " + " rem=" + removed.size() + " "
							+ (System.currentTimeMillis() - t) + "ms");
		}
	}

	private final void addSample(TrainingSample<T> t) {
		double z = valueOf(t.sample);
		double yz = z * t.label;
		if (yz < 1 && yz + C * kernel.valueOf(t.sample, t.sample) > 0) {
			synchronized (train) {
				if (!train.contains(t)) {
					SV sv = new SV(t, 0, z);
					train.add(sv);
					updateNoCache(train.size() - 1);
				} else {
					updateNoCache(train.indexOf(t));
				}
				if (train.size() > capacity * budget) {
					prune();
				}
			}
		}
	}

	/**
	 * dual variable update
	 * 
	 * @param i
	 *            index f the dual variable
	 */
	private final void updateNoCache(int i) {
		synchronized (train) {
			SV sv = train.get(i);
			double y = sv.sample.label;
			double z = sv.z;
			if (y * z != 1) {
				double preva = sv.alpha;
				double da = (1 - y * z)
						/ kernel.valueOf(sv.sample.sample, sv.sample.sample);
				sv.alpha = y * max(0, min(C, da + y * sv.alpha));
				double DA = sv.alpha - preva;
				if (sv.alpha == 0) {
					train.remove(i);
				}
				for (SV s : train) {
					double k = kernel
							.valueOf(sv.sample.sample, s.sample.sample);
					s.z += DA * k;
				}
			}
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.classifier.Classifier#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(T e) {
		double z = 0;
		synchronized (train) {
			for (SV sv : train) {
				z += sv.alpha * kernel.valueOf(sv.sample.sample, e);
			}
		}
		return z;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see fr.lip6.jkernelmachines.classifier.Classifier#copy()
	 */
	@SuppressWarnings("unchecked")
	@Override
	public Classifier<T> copy() throws CloneNotSupportedException {
		return (BudgetSDCA<T>) this.clone();
	}

	/**
	 * Get the number of epochs to train the classifier
	 * 
	 * @return the number of epochs
	 */
	public int getE() {
		return E;
	}

	/**
	 * Set the number of epochs (going through all samples once) for the
	 * training phase
	 * 
	 * @param e
	 *            the number of epochs
	 */
	public void setE(int e) {
		E = e;
	}

	@Override
	public void setKernel(Kernel<T> k) {
		this.kernel = k;
	}

	@Override
	public double[] getAlphas() {
		double[] a = new double[train.size()];
		for (int s = 0; s < a.length; s++) {
			SV sv = train.get(s);
			a[s] = sv.alpha * sv.sample.label;
		}
		return a;
	}

	@Override
	public void setC(double c) {
		C = c;
	}

	@Override
	public double getC() {
		return C;
	}

	@Override
	public Kernel<T> getKernel() {
		return kernel;
	}

	public int getBudget() {
		return budget;
	}

	public void setBudget(int budget) {
		this.budget = budget;
	}

	public double getCapacity() {
		return capacity;
	}

	public void setCapacity(double capacity) {
		this.capacity = capacity;
	}
}
