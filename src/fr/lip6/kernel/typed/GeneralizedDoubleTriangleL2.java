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
package fr.lip6.kernel.typed;

import java.util.List;

import fr.lip6.kernel.Kernel;
import fr.lip6.threading.ThreadedMatrixOperator;
import fr.lip6.type.TrainingSample;

/**
 * Triangular Kernel on double[] that uses a Generalized L2 distance.
 * @author dpicard
 *
 */
public class GeneralizedDoubleTriangleL2 extends Kernel<double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1626829154456556731L;
	
	private double[] gammas;
	
	/**
	 * Constructor using an array of weighted for the generalized L2 distance
	 * @param gamma the array of weights
	 */
	public GeneralizedDoubleTriangleL2(double[] gamma)
	{
		this.gammas = gamma;
	}
	
	@Override
	public double valueOf(double[] t1, double[] t2) {
		
		if(t1.length != gammas.length || t2.length!= gammas.length)
		{
			System.err.println("not same length t1 : "+t1.length+" t2 : "+t2.length+" gamma : "+gammas.length);
			return -1;
		}
		
		double sum = 0.;
		double tmp = 0.;
		for (int i = 0; i < Math.min(t1.length, t2.length); i++)
			//assume X and Y > 0
			if( gammas[i] != 0)
			{
				tmp = t1[i] - t2[i];
				sum += gammas[i] * tmp*tmp; //chi2
			}
		
//		return Math.max(0, 1 - sum);
		return 1 - sum;
	}

	@Override
	public double valueOf(double[] t1) {
		return 1.0;
	}


	/**
	 * @return the sigma
	 */
	public double[] getGammas() {
		return gammas;
	}

	/**
	 * @param gamma inverse of std dev parameter
	 */
	public void setGammas(double[] gamma) {
		this.gammas = gamma;
	}
	
	public double distanceValueOf(double[] t1, double[] t2) {

		
		if(t1.length != gammas.length || t2.length!= gammas.length)
		{
			System.err.println("not same length t1 : "+t1.length+" t2 : "+t2.length+" gamma : "+gammas.length);
			return -1;
		}
		
		double sum = 0.;
		double tmp = 0.;
		for (int i = 0; i < Math.min(t1.length, t2.length); i++)
			//assume X and Y > 0
			if( gammas[i] != 0)
			{
				tmp = t1[i] - t2[i];
				sum += gammas[i] * tmp*tmp; //chi2
			}
		
		return sum;
	}
	
	public double[][] distanceMatrix(final List<TrainingSample<double[]>> l, final int x)
	{
		double[][] matrix = new double[l.size()][l.size()];
		
//		if(gammas[x] == 0)
//			return matrix;
		//computing matrix				
		ThreadedMatrixOperator factory = new ThreadedMatrixOperator()
		{
			@Override
			public void doLines(double[][] matrix, int from, int to) {
				double tmp = 0;
				for(int index = from ; index < to ; index++)
				{
					double s1 = l.get(index).sample[x];
					for(int j = 0 ; j < matrix.length ; j++){
						tmp = s1 - l.get(j).sample[x];
//						matrix[index][j] = gammas[x]*tmp*tmp;
						matrix[index][j] = tmp*tmp;
					}
						
				}
			}
		};
		
		factory.getMatrix(matrix);
		
		return matrix;
	}
}
