from examples.numerical_examples.experiment import alt_experiment, print_results
from examples.numerical_examples.performance_functions import portfolio_low,portfolio_high



param_dict = {
    'd': [32],
    'N': [2000],
    'max_it': [30],
    'k_init':[1],
    'CV_target': [1.5]
}

filename='ice_portfolio_low.pkl'

alt_experiment(param_dict=param_dict,
               performance_function=portfolio_low,
               seeds=100,
               filename=filename,
               algo='ice')

params = list(param_dict.keys())

print_results(filename,params,False)




################################################


param_dict = {
    'd': [102,252],
    'N': [2000],
    'max_it': [30],
    'k_init':[1],
    'CV_target': [1.5]
}

filename='ice_portfolio_high.pkl'

alt_experiment(param_dict=param_dict,
               performance_function=portfolio_high,
               seeds=100,
               filename=filename,
               algo='ice')

params = list(param_dict.keys())

print_results(filename,params,False)