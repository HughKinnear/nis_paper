import pickle


per_funcs = ['pwl','meatball','two_dof','suspension']


d = 'examples/numerical_examples/'

for func in per_funcs:
    filename = d + func + '/' + 'dmc_' + func + '.pkl'
    with open(filename, "rb") as file:
        results = pickle.load(file)
    print(func)
    print(results)

filename = 'examples/numerical_examples/portfolio/dmc_portfolio_32.pkl'
with open(filename, "rb") as file:
    results = pickle.load(file)
print('portfolio_32')
print(results)

filename = 'examples/numerical_examples/portfolio/dmc_portfolio_102.pkl'
with open(filename, "rb") as file:
    results = pickle.load(file)
print('portfolio_102')
print(results)

filename = 'examples/numerical_examples/portfolio/dmc_portfolio_252.pkl'
with open(filename, "rb") as file:
    results = pickle.load(file)
print('portfolio_252')
print(results)






