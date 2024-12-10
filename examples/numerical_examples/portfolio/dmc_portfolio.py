from examples.numerical_examples.experiment import DirectMonteCarlo
from examples.numerical_examples.performance_functions import portfolio_low,portfolio_high
import pickle


##################################################################

size = 10**7
seed = 0

dmc = DirectMonteCarlo(dim=32,
                       performance_function=portfolio_low,
                       sample_size=size,
                       chunks=100,
                       is_sample_save=False,
                       seed=seed)

dmc.compute()
estimate = dmc.threshold(0)

results = (estimate,size,seed)
filename = 'dmc_portfolio_32.pkl'

with open(filename, 'wb') as file:
    pickle.dump(results, file)


##################################################################

size = 10**7
seed = 0

dmc = DirectMonteCarlo(dim=102,
                       performance_function=portfolio_high,
                       sample_size=size,
                       chunks=100,
                       is_sample_save=False,
                       seed=seed)

dmc.compute()
estimate = dmc.threshold(0)

results = (estimate,size,seed)
filename = 'dmc_portfolio_102.pkl'

with open(filename, 'wb') as file:
    pickle.dump(results, file)


##################################################################

size = 10**7
seed = 0

dmc = DirectMonteCarlo(dim=252,
                       performance_function=portfolio_high,
                       sample_size=size,
                       chunks=100,
                       is_sample_save=False,
                       seed=seed)

dmc.compute()
estimate = dmc.threshold(0)

results = (estimate,size,seed)
filename = 'dmc_portfolio_252.pkl'

with open(filename, 'wb') as file:
    pickle.dump(results, file)


