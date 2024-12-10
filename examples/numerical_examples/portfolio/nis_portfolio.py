from examples.numerical_examples.experiment import nis_experiment, print_results
from examples.numerical_examples.performance_functions import portfolio_high, portfolio_low

########################################################################

param_dict = {
    'dimension': [32],
    'fitting_multiplier': [30]
}

filename='nis_portfolio_low.pkl'

nis_experiment(param_dict=param_dict,
               performance_function=portfolio_low,
               seeds=100,
               filename=filename)

params = list(param_dict.keys())
print_results(filename,params,True)

########################################################################

param_dict = {
    'dimension': [102,252],
    'fitting_multiplier': [30]
}

filename='nis_portfolio_high.pkl'

nis_experiment(param_dict=param_dict,
               performance_function=portfolio_high,
               seeds=100,
               filename=filename)

params = list(param_dict.keys())
print_results(filename,params,True)

########################################################################


