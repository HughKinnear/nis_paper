from examples.numerical_examples.experiment import alt_experiment, print_results
from examples.numerical_examples.performance_functions import c_suspension

param_dict = {
    'd': [3,99,300],
    'N': [2000],
    'max_it': [30],
    'k_init':[1],
    'CV_target': [1.5]
}

filename='ice_suspension.pkl'

alt_experiment(param_dict=param_dict,
               performance_function=c_suspension,
               seeds=100,
               filename=filename,
               algo='ice')

params = list(param_dict.keys())

print_results(filename,params,False)