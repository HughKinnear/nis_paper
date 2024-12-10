from examples.numerical_examples.experiment import DirectMonteCarlo
from examples.numerical_examples.performance_functions import two_dof
import pickle

size = 10**7
seed = 0

dmc = DirectMonteCarlo(dim=2,
                       performance_function=two_dof,
                       sample_size=size,
                       chunks=100,
                       is_sample_save=False,
                       seed=seed)

dmc.compute()
estimate = dmc.threshold(0)

results = (estimate,size,seed)
filename = 'dmc_two_dof.pkl'

with open(filename, 'wb') as file:
    pickle.dump(results, file)

