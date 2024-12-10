from examples.numerical_examples.experiment import DirectMonteCarlo
from examples.numerical_examples.performance_functions import suspension 
import pickle

size = 10**8
seed = 0

dmc = DirectMonteCarlo(dim=3,
                       performance_function=suspension,
                       sample_size=size,
                       chunks=100,
                       is_sample_save=False,
                       seed=seed)

dmc.compute()
estimate = dmc.threshold(0)

results = (estimate,size,seed)
filename = 'dmc_suspension.pkl'

with open(filename, 'wb') as file:
    pickle.dump(results, file)
