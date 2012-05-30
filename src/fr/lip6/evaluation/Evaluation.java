package fr.lip6.evaluation;

import java.io.Serializable;


/**
 * Simple class containing a sample and its evaluation by the classifier
 * @author dpicard
 *
 * @param <T> sample data type
 */
public class Evaluation<T>  implements Comparable<Evaluation<T>>, Serializable
{
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 791024170617779718L;
	
	T sample;
	double value;
	
	public Evaluation(T s, double v)
	{
		sample = s;
		value = v;
	}
	
	@Override
	public int compareTo(Evaluation<T> o) {
		if(o == null)
			return 0;
		return (int) Math.signum(o.value - value);
	}
}