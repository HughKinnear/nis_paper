import numpy as np
from .importance_sample import ReliabilityImportanceSampler
from .performance_function import PerformanceFunction
from .markov_chain import RecycleChainData, RecycleModifiedMetropolis
from .initial_sample import InitialSampler
from .distribution import vMFNM
from scipy.stats import entropy
from .em import EMvMFNM


class NichingImportanceSampling:

    def __init__(self,
                 performance_function,
                 dimension,
                 seed,
                 fitting_multiplier,
                 cov_target=0.1,
                 weight_cov_target=5,
                 importance_step=250,
                 scale=0.8,
                 threshold=0,
                 converge_limit=20,
                 level_probability=0.1,
                 max_initial_samples=10,
                 max_evals=100000,
                 max_len=100,
                 attempt_sequence=None,
                 minimum_dimension=25,
                 distribution=vMFNM):
        self.dimension = dimension
        self.fitting_multiplier = fitting_multiplier
        self.cov_target = cov_target
        self.weight_cov_target = weight_cov_target
        self.threshold = threshold
        self.max_len = max_len
        self.importance_step = importance_step
        self.distribution = distribution
        self.minimum_dimension = minimum_dimension
        self.max_evals = max_evals
        self.scale = scale
        self.attempt_sequence = (list(np.arange(0,4,0.04))
                                 if attempt_sequence is None else attempt_sequence)
        self.converge_limit = converge_limit
        self.level_probability = level_probability
        self.max_initial_samples = max_initial_samples
        self.performance_function = PerformanceFunction(performance_function)
        self.random_state = np.random.default_rng(seed)
        self.indicator = lambda x : int(self.performance_function(x) >= self.threshold)
        self.markov_chain = RecycleModifiedMetropolis(scale=scale,
                                                      random_state=self.random_state)   
        self.initial_sampler = InitialSampler(attempt_sequence=self.attempt_sequence,
                                              performance_function=self.performance_function,
                                              threshold=threshold,
                                              dimension=dimension,
                                              max_len=max_len,
                                              random_state=self.random_state,
                                              converge_limit=converge_limit,
                                              level_probability=level_probability,
                                              max_initial_samples=max_initial_samples,
                                              markov_chain=self.markov_chain)
        self.mixture_chain_data = None
        self.chain_weights = None
        self.importance_samplers = []
        self.step_completed = 0
        self.estimates = []
        self.covs = []
        self.weight_covs = []
        self.eff_niches = []
        self.is_results = []


    def run(self):
        self.run_initial_sampler()
        self.print_table_header()
        while not self.is_eval_stop and not self.is_cov_stop:
            if self.is_weight_cov_large:
                self.mcmc_update()
                self.fit()        
            self.importance_sampler.sample(self.importance_step)
            self.compute_stats()
            self.print_table_row()
        self.print_evals()

    def run_initial_sampler(self):
        self.initial_sampler.run()
        self.performance_function.save('Initial Sample')
        self.mixture_chain_data = [RecycleChainData(accept_chain_list=[[sample.array]],
                                                    propose_chain_list=[[]])
                                   for sample in self.initial_sampler.initial_samples]
        n = len(self.mixture_chain_data)
        self.chain_weights = np.array([1/n]*n)

    def mcmc_update(self):
        eff_niche = 1 if self.step_completed == 0 else self.eff_niche
        dim = max(self.dimension,self.minimum_dimension)
        total_budget = self.fitting_multiplier * eff_niche * dim
        budget_list = [int(val) for val in total_budget * self.chain_weights]
        for chain_data,budget in zip(self.mixture_chain_data,
                                     budget_list):
            self.markov_chain.update(chain_data, budget,self.indicator)
        self.performance_function.save('Markov Chain')


    def fit(self):
        data_list = [chain_data.all_accept_samples for chain_data in self.mixture_chain_data]
        chain_lens = [len(data) for data in data_list]

        X = np.vstack(data_list).T
        W = np.ones((len(X.T),1))
        M = self.construct_m(chain_lens)

        [mu, kappa, m, omega, pi] = EMvMFNM(X, W, M)
        random_state = np.random.default_rng(0)
        dist = vMFNM(pi.T,
                    mu.T,
                    kappa.T,
                    omega.T,
                    m.T,
                    random_state)

        importance_sampler = ReliabilityImportanceSampler(dist,
                                                        self.performance_function,
                                                        0,
                                                        random_state)
        importance_sampler.importance_samples = X.T
        results = importance_sampler.compute_stats()
        weights = results.cross_entropy_weights
        weights[weights < 1e-10] = 0
        weights = weights / np.sum(weights)
        dist.weights = weights
        importance_values = results.importance_values
        self.chain_weights = np.sum(M * importance_values[:,None],axis=0) / np.sum(importance_values)

        self.importance_samplers.append(ReliabilityImportanceSampler(dist,
                                                        self.performance_function,
                                                        0,
                                                        random_state))

    @staticmethod
    def construct_m(k_values):
        n = sum(k_values)
        k = len(k_values)
        result = np.zeros((n, k), dtype=int)
        
        start_index = 0
        for col, count in enumerate(k_values):
            result[start_index:start_index+count, col] = 1
            start_index += count
        return result
    
    def compute_stats(self):
        self.step_completed += 1
        results = self.importance_sampler.compute_stats()
        self.is_results.append(results)
        self.estimates.append(results.estimate)
        cov_estimate = np.inf if np.isnan(results.cov_estimate) else results.cov_estimate
        self.covs.append(cov_estimate)
        weight_cov_estimate = np.inf if np.isnan(results.weight_cov_estimate) else results.weight_cov_estimate
        self.weight_covs.append(weight_cov_estimate)
        self.eff_niches.append(self.compute_eff_niche(results.posterior_probs))
        self.performance_function.save('Importance Sample')

    @staticmethod
    def compute_eff_niche(posterior_probs):
        total_entropy = entropy(np.mean(posterior_probs,axis=0))
        overlap_entropy = np.mean(entropy(posterior_probs,axis=1))
        return np.exp(total_entropy - overlap_entropy)
    
    
    @property
    def is_eval_stop(self):
        if not self.covs:
            return False
        return self.performance_function.eval_count >= self.max_evals
    
    @property
    def is_weight_cov_large(self):
        if not self.weight_covs:
            return True
        return self.weight_cov > self.weight_cov_target
    
    @property
    def is_cov_stop(self):
        if not self.covs:
            return False
        return self.cov <= self.cov_target
    
    @property
    def importance_sampler(self):
        return self.importance_samplers[-1]
    
    @property
    def estimate(self):
        return self.estimates[-1]
    
    @property
    def cov(self):
        return self.covs[-1]
    
    @property
    def eff_niche(self):
        return self.eff_niches[-1]
    
    @property
    def weight_cov(self):
        return self.weight_covs[-1]
    
    @property
    def eff_comp(self):
        return self.eff_comps[-1]

    def effective_number_of_components(self,weights):
        sum_of_squares = sum(w ** 2 for w in weights)
        n_eff = 1 / sum_of_squares if sum_of_squares > 0 else 0
        return n_eff
    
    def fit_is(self,dist):
        return ReliabilityImportanceSampler(dist,
                                            self.performance_function,
                                            self.threshold,
                                            self.random_state)

    def print_table_header(self):
        header_str = (
            f'{'Step':<10} '
            f'{'Estimate':<12} '
            f'{'CoV':<10} '
            f'{'Weight CoV':<12} '
            f'{'Eff Niches':<12} '
        )
        print()
        print()
        print(header_str)
        print("-" * 60)  

    def print_table_row(self):
        formatted_estimate = f'{format(self.estimate, ".2e"):<12}'
        row_str = (
            f'{self.step_completed:<10} '
            f'{formatted_estimate} '
            f'{round(self.cov,2):<10} '
            f'{round(self.weight_cov,2):<12} '
            f'{round(self.eff_niche,2):<12} '
        )
        print(row_str)

    def print_evals(self):
        print()
        print('Performance Function Evaluations')
        print('---------------------------------')
        my_dict = self.performance_function.eval_count_dict
        max_key_length = max(len(key) for key in my_dict)
        for key, value in my_dict.items():
            print(f"{key:{max_key_length}} : {value}")
        

