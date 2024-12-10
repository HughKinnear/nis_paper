from examples.numerical_examples.experiment import alt_experiment, print_results
from examples.numerical_examples.performance_functions import c_two_dof

param_dict = {
    'd': [2,100,300],
    'N': [2000],
    'p': [0.1],
    'burn' : [0],
    'tarCOV': [1.5]
}

filename='sis_two_dof.pkl'

alt_experiment(param_dict=param_dict,
               performance_function=c_two_dof,
               seeds=100,
               filename=filename,
               algo='sis')

params = list(param_dict.keys())

print_results(filename,params,False)