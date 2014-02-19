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

    Copyright David Picard - 2014

 */
package fr.lip6.jkernelmachines.density;

import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.sqrt;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import fr.lip6.jkernelmachines.util.DebugPrinter;
import fr.lip6.jkernelmachines.util.algebra.MatrixOperations;
import fr.lip6.jkernelmachines.util.algebra.MatrixVectorOperations;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

/**
 * @author picard
 * 
 */
public class DoubleGaussianMixtureModel implements DensityFunction<double[]> {

	int k;
	double[] pop;
	double[][] mu;
	double[][][] sigma;
	
	DebugPrinter debug = new DebugPrinter();

	public DoubleGaussianMixtureModel(int nb_gaussian) {
		k = nb_gaussian;
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.density.DensityFunction#train(java.lang.Object)
	 */
	@Override
	public void train(double[] e) {
		// TODO Auto-generated method stub

	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.density.DensityFunction#train(java.util.List)
	 */
	@Override
	public void train(List<double[]> e) {
		int n = e.size();
		int dim = e.get(0).length;

		pop = new double[k];
		mu = new double[k][dim];
		sigma = new double[k][dim][dim];

		double[] oldmu = new double[dim];
		
		// init
		double[][] like = new double[n][k];
		
		Random rand = new Random();
		/*
		for (int x = 0; x < n; x++) {
			int d = rand.nextInt(k);
			like[x][d] = 1.0;
		}
		for (int g = 0; g < k; g++) {
			pop[g] = 0;
			Arrays.fill(mu[g], 0);
			for (int d = 0; d < dim; d++) {
				Arrays.fill(sigma[g][d], 0);
			}
		}
		// M
		for (int x = 0; x < n; x++) {
			double[] xt = e.get(x);
			for (int g = 0; g < k; g++) {
				pop[g] += like[x][g];
				VectorOperations.addi(mu[g], mu[g], like[x][g], xt);
			}
		}
		for (int g = 0; g < k; g++) {
			VectorOperations.muli(mu[g], mu[g], 1. / pop[g]);
		}
		for (int x = 0; x < n; x++) {
			double[] xt = e.get(x);
			for (int g = 0; g < k; g++) {
				double[] xtc = VectorOperations.add(xt, -1, mu[g]);
				VectorOperations.muli(xtc, xtc, sqrt(like[x][g]));
				MatrixVectorOperations.addXXTrans(sigma[g], xtc);
			}
		}
		for (int g = 0; g < k; g++) {
			for (int d = 0; d < dim; d++) {
				VectorOperations.muli(sigma[g][d], sigma[g][d], 1. / pop[g]);
			}
			sigma[g] = MatrixOperations.inv(sigma[g]);
		}
		VectorOperations.muli(pop, pop, 1./n);
		*/
		

		int t = 0;
		// init with k-means
		int c[] = new int[n];
		for(int i = 0 ; i < n ; i++) {
			c[i] = rand.nextInt(k);
			VectorOperations.addi(mu[c[i]], mu[c[i]], 1, e.get(i));
			pop[c[i]] += 1;
		}
		for(int g = 0 ; g < k ; g++) {
			VectorOperations.muli(mu[g], mu[g], 1./pop[g]);
		}
		
		for(; t < 100 ; t++) {
			boolean cont = false;
			// E
			for(int i = 0 ; i < n ; i++){
				double[] x = e.get(i);
				double dmin = Double.POSITIVE_INFINITY;
				int cmin = -1;
				for(int g = 0 ; g < k ; g++) {
					double d = VectorOperations.d2p2(x, mu[g]);
					if(d < dmin) {
						cmin = g;
						dmin = d;
					}
				}
				if(cmin != c[i])
					cont = true;
				c[i] = cmin;
			}
			
			if(!cont)
				break;
			
			// M
			for(int g = 0 ; g < k ; g++) {
				Arrays.fill(mu[g], 0);
				pop[g] = 0;
			}
			for(int i = 0 ; i < n ; i++) {
				VectorOperations.addi(mu[c[i]], mu[c[i]], 1, e.get(i));
				pop[c[i]] += 1;
			}
			for(int g = 0 ; g < k ; g++) {
				VectorOperations.muli(mu[g], mu[g], 1./pop[g]);
			}
		}
		
		debug.println(3, "init k-means :");
		debug.println(3, " t = "+t);
		for (int i = 0; i < k; i++)
			debug.println(3, " mu"+i+" = "+Arrays.toString(mu[i]));
		
		// diag cov
		for(int g = 0 ; g < k ; g++) {
			pop[g] /= n;
			for(int d = 0 ; d < dim ; d++) {
				sigma[g][d][d] = 1.0;
			}
		}
		for (t = 0 ; t < 1000; t++) {
			// E
			for (int x = 0; x < n; x++) {
				double[] xt = e.get(x);
				double sum = 0;
				for (int g = 0; g < k; g++) {
					double[] xtc = VectorOperations.add(xt, -1, mu[g]);
					double[] o = MatrixVectorOperations.rMul(sigma[g], xtc);
					like[x][g] = (pop[g]) * exp((-0.5*VectorOperations.dot(xtc, o)));
					if(like[x][g] < 1e-15)
						like[x][g] = 0;
					sum += like[x][g];
				}
				if (abs(sum) > 1e-15)
					VectorOperations.muli(like[x], like[x], 1. / sum);
			}
			// M
			for (int g = 0; g < k; g++) {
				pop[g] = 0;
				Arrays.fill(mu[g], 0);
				for (int d = 0; d < dim; d++) {
					Arrays.fill(sigma[g][d], 0);
				}
			}
			// weights and means
			for (int x = 0; x < n; x++) {
				double[] xt = e.get(x);
				for (int g = 0; g < k; g++) {
					pop[g] += like[x][g];
					VectorOperations.addi(mu[g], mu[g], like[x][g], xt);
				}
			}
			for (int g = 0; g < k; g++) {
				VectorOperations.muli(mu[g], mu[g], 1. / pop[g]);
			}
			// covariances
			for (int x = 0; x < n; x++) {
				double[] xt = e.get(x);
				for (int g = 0; g < k; g++) {
					double[] xtc = VectorOperations.add(xt, -1, mu[g]);
					VectorOperations.muli(xtc, xtc, sqrt(like[x][g]));
					MatrixVectorOperations.addXXTrans(sigma[g], xtc);
				}
			}
			for (int g = 0; g < k; g++) {
				for (int d = 0; d < dim; d++) {
					VectorOperations.muli(sigma[g][d], sigma[g][d], 1. / (2*pop[g]));
				}
				sigma[g] = MatrixOperations.inv(sigma[g]);
			}
			// normalize weights
			VectorOperations.muli(pop, pop, 1./n);
			
			// if no improvements, quit
			if(VectorOperations.d2p2(mu[0], oldmu) < 1e-12)
				break;
			oldmu = Arrays.copyOf(mu[0], dim);

		}
		debug.println(3, "t = "+t);
		debug.println(3, "pop= " + Arrays.toString(pop));
		
		for (int i = 0; i < k; i++)
			debug.println(3, "mu"+i+" = "+Arrays.toString(mu[i]));
		
		for (int i = 0; i < k; i++) {
			debug.println(3, "sigma"+i+" = "+Arrays.deepToString(sigma[i]));
		}
	}

	/*
	 * (non-Javadoc)
	 * 
	 * @see
	 * fr.lip6.jkernelmachines.density.DensityFunction#valueOf(java.lang.Object)
	 */
	@Override
	public double valueOf(double[] e) {
		double sum = 0;
		for (int g = 0; g < k; g++) {
			double[] xtc = VectorOperations.add(e, -1, mu[g]);
			double[] o = MatrixVectorOperations.rMul(sigma[g], xtc);
			sum += pop[g] * exp(-0.5*VectorOperations.dot(xtc, o));

		}
		return sum;
	}

}
