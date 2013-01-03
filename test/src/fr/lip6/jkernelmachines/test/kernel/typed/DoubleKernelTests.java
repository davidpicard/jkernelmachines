package fr.lip6.jkernelmachines.test.kernel.typed;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

@RunWith(Suite.class)
@SuiteClasses({ DoubleGaussChi1Test.class, DoubleGaussChi2Test.class, DoubleGaussL2Test.class,
		DoubleLinearTest.class })
public class DoubleKernelTests {

}
