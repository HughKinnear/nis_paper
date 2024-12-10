from examples.sis.implementation import sis
from examples.ice_vmfnm.implementation import ice_vmfnm
from tqdm.auto import tqdm
import numpy as np
from nis.nis import NichingImportanceSampling
import pickle
import contextlib
from itertools import product
from nis.performance_function import PerformanceFunction
from scipy.stats import multivariate_normal
import gc


def nis_experiment(param_dict,performance_function,seeds,filename):
    results = []
    keep_keys = ['dimension','fitting_multiplier','cov_target', 'weight_cov_target',
        'threshold','importance_step','minimum_dimension','max_evals',
        'scale','converge_limit','attempt_sequence',
        'level_probability','max_initial_samples','max_len']
    iterator = list(product(*param_dict.values()))
    for values in tqdm(iterator,desc='Overall'):
        params = dict(zip(param_dict.keys(), values))
        result_dict = {
            'seeds':list(range(seeds)),
            'estimates': [],
            'evals': []
        }
        for seed in tqdm(range(seeds),desc='Experiment'):
            gc.collect()
            nis = NichingImportanceSampling(seed=seed,
                                            performance_function=performance_function,
                                            **params)
            try:
                with contextlib.redirect_stdout(None):
                    nis.run()
                result_dict['estimates'].append(nis.estimate)
                result_dict['evals'].append(nis.performance_function.eval_count_dict)
            except Exception as e:
                print(f"An error occurred: {e}")
                print(seed)
                print(params)
        for keep_key in keep_keys:
            result_dict[keep_key] = vars(nis)[keep_key]
        results.append(result_dict)
        with open(filename, 'wb') as file:
            pickle.dump(results, file)

def alt_experiment(param_dict,performance_function,seeds,filename,algo):
    algo = ice_vmfnm if algo == 'ice' else sis
    results = []
    iterator = list(product(*param_dict.values()))
    for values in tqdm(iterator,desc='Overall'):
        params = dict(zip(param_dict.keys(), values))
        result_dict = params.copy()
        result_dict['seeds'] = list(range(seeds))
        result_dict['estimates'] = []
        result_dict['evals'] = []
        for seed in tqdm(range(seeds),desc='Experiment'):
            gc.collect()
            g = PerformanceFunction(performance_function)
            try:
                with contextlib.redirect_stdout(None):
                    exp_results = algo(seed=seed,g=g,**params)
                result_dict['estimates'].append(exp_results.failure_probability)
                result_dict['evals'].append(g.eval_count)
            except Exception as e:
                print(f"An error occurred: {e}")
                print(seed)
                print(params)          
        results.append(result_dict)
        with open(filename, 'wb') as file:
            pickle.dump(results, file)
            

class NisExperimentResult:

    def __init__(self,result):
        for key, value in result.items():
            setattr(self, key, value)
        self.mean_estimate = np.mean(self.estimates)
        self.std_estimate = np.std(self.estimates)
        self.cov_estimate = self.std_estimate / self.mean_estimate

        keys = self.evals[0].keys()
        values = np.array([list(d.values()) for d in self.evals])
        mean_values = np.mean(values, axis=0)
        self.mean_evals = dict(zip(keys, mean_values))

    def print(self,params):
        for param in params:
            print(f'{param:<20}: {getattr(self,param)}')
        print('-'*30)
        print(f'{'Mean Estimate':<20}: {self.mean_estimate:.2e}')
        print(f'{'CoV':<20}: {round(self.cov_estimate,2)}')
        print(f'{'Mean Evals':<20}: {self.mean_evals['Total']:.2e}')

class AltExperimentResult:

    def __init__(self,result):
        for key, value in result.items():
            setattr(self, key, value)
        self.mean_estimate = np.mean(self.estimates)
        self.std_estimate = np.std(self.estimates)
        self.cov_estimate = self.std_estimate / self.mean_estimate
        self.mean_evals = np.mean(self.evals)

    def print(self,params):
        for param in params:
            print(f'{param:<20}: {getattr(self,param)}')
        print('-'*30)
        print(f'{'Mean Estimate':<20}: {self.mean_estimate:.2e}')
        print(f'{'CoV':<20}: {round(self.cov_estimate,2)}')
        print(f'{'Mean Evals':<20}: {self.mean_evals:.2e}')


def print_results(filename,params,is_nis):
    result_class = NisExperimentResult if is_nis else AltExperimentResult
    with open(filename, "rb") as file:
        results = pickle.load(file)
    exp_results = [result_class(result) for result in results]
    for exp_result in exp_results:
        print('='*50)
        exp_result.print(params)
        print('='*50)



class DirectMonteCarlo:

    def __init__(self,
                 dim,
                 performance_function,
                 sample_size,
                 chunks,
                 is_sample_save,
                 seed):
        self.dim = dim
        self.performance_function = performance_function
        self.sample_size = sample_size
        self.seed = seed
        self.random_state = np.random.default_rng(seed)
        self.chunks = chunks
        self.chunk_size = sample_size // chunks
        self.performances = []
        self.is_sample_save = is_sample_save
        if self.is_sample_save:
            self.samples = []
    
    def compute(self):
        for _ in tqdm(range(self.chunks)):
            samples = multivariate_normal.rvs(mean=np.zeros(self.dim),
                                               random_state=self.random_state,
                                               size=self.chunk_size)
            for sample in samples:
                self.performances.append(self.performance_function(sample))
                if self.is_sample_save:
                    self.samples.append(sample)

    def threshold(self,threshold):
        return sum(np.array(self.performances) >= threshold) / self.sample_size
