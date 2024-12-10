from examples.numerical_examples.experiment import nis_experiment, print_results
from examples.numerical_examples.performance_functions import c_pwl

param_dict = {
    'dimension': [2,100,300],
    'fitting_multiplier': [30],
}

filename='nis_pwl.pkl'

nis_experiment(param_dict=param_dict,
               performance_function=c_pwl,
               seeds=100,
               filename=filename)

params = list(param_dict.keys())
print_results(filename,params,True)
