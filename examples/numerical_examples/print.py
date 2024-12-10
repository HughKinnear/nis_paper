from examples.numerical_examples.experiment import print_results
from itertools import product


per_funcs = ['pwl','meatball','two_dof','suspension']
methods = ['sis','ice','nis']


d = 'examples/numerical_examples/'

for pair in product(per_funcs,methods):
    print(pair)
    filename = d + pair[0] + '/' + pair[1] + '_' + pair[0] + '.pkl'
    print_results(filename,[],pair[1]=='nis')

type = ['low', 'high']


for pair in product(type,methods):
    print(pair)
    filename = d + 'portfolio/' + pair[1] + '_portfolio_' + pair[0] + '.pkl'
    print_results(filename,[],pair[1]=='nis')






