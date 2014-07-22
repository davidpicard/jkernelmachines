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

package fr.lip6.jkernelmachines.gui;

import fr.lip6.jkernelmachines.classifier.Classifier;
import fr.lip6.jkernelmachines.projection.DoublePCA;
import java.io.Serializable;

/**
 * Very simple model class that encapsulate a classifier and preprocessing tools
 * 
 * @author David Picard
 */
public class Model implements Serializable {

	private static final long serialVersionUID = 1L;
	int dim;
	boolean pcaEnable;
	boolean whiteningEnable;
	boolean normalizationEnable;
	DoublePCA pca;
	Classifier<double[]> classifier;
}
