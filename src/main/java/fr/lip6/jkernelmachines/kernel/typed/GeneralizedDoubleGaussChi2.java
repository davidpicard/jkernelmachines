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
package fr.lip6.jkernelmachines.kernel.typed;

import fr.lip6.jkernelmachines.kernel.Kernel;

/**
 * Gaussian Kernel on double[] that uses a custom weighted Chi2 distance.
 * @author dpicard
 *
 */
public class GeneralizedDoubleGaussChi2 extends Kernel<double[]> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1626829154456556731L;
	
	
	private double eps = 10e-6;
	private double[] gammas;
	private double gamma = 1.0;
	
	/**
	 * Constructor with the array of weighted for the distance
	 * @param gamma the array of weights for each component
	 */
	public GeneralizedDoubleGaussChi2(double[] gamma)
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
		double min = 0.;
		for (int i = 0; i < Math.min(t1.length, t2.length); i++)
			//assume X and Y > 0
			if( (tmp = t1[i]+t2[i]) > eps && gammas[i] != 0)
			{
				min = t1[i]-t2[i];
				sum += gammas[i] * (min*min) / tmp; //chi2
			}
		
		return Math.exp(-gamma * sum);
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


	/**
	 * @param gamma inverse of std dev parameter
	 */
	public void setGamma(double gamma) {
		this.gamma = gamma;
	}
}
