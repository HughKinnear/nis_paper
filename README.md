# Niching Importance Sampling

This is a companion repository for the paper Niching importance sampling. It has two purposes.

The first is reproducibility. All the code for the numerical experiments (examples/numerical_examples) and figures (examples/figures) used in the paper are shared here. The file examples/numerical_examples/print.py prints the results of all numerical experiments conducted.

The second is to provide an implementation of niching importance sampling that others can use on their own problems. The file nis/nis.py contains the class NichingImportanceSampling that can be applied to any performance function.