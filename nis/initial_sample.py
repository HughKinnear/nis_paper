import numpy as np
from dataclasses import dataclass
from scipy.stats import norm
import matplotlib.pyplot as plt
from .markov_chain import RecycleChainData


@dataclass
class Sample:
    array: ...
    performance: ...


class SingleChain:

    def __init__(self,
                 sample_list):
        self.sample_list = sample_list
        self.sorted_samples = sorted(sample_list,key=lambda x: x.performance)

    @property
    def child_threshold(self):
        return self.sorted_samples[-1].performance
    
    @property
    def child_seed(self):
        return self.sorted_samples[-1]
    
    
class ChainRun:

    def __init__(self):
        self.chains = []

    @property
    def current_chain(self):
        return self.chains[-1]
    
    @property
    def all_samples(self):
        return [sample for chain in self.chains for sample in chain.sample_list]
    
    def add(self, chain):
        self.chains.append(chain)

    def threshold_samples(self,threshold):
        return [samp for samp in self.current_chain.sample_list
                if samp.performance >= threshold]
    
    def plot(self):
        plotter = np.array([samp.array for samp in self.all_samples])
        plt.scatter(plotter[:,0],plotter[:,1])


class InitialSampler:

    def __init__(self,
                 attempt_sequence,
                 converge_limit,
                 performance_function,
                 threshold,
                 random_state,
                 dimension,
                 level_probability,
                 max_initial_samples,
                 max_len,
                 markov_chain):
        self.converge_limit = converge_limit
        self.threshold = threshold
        self.dimension = dimension
        self.max_len = max_len
        self.level_probability = level_probability
        self.max_initial_samples = max_initial_samples
        self.chain_len = int(level_probability ** -1)
        self.random_state = random_state
        self.performance_function = performance_function
        self.markov_chain = markov_chain
        self.attempt_sequence = attempt_sequence
        
        self.representatives = []
        self.initial_samples = []
        self.chain_runs = []
        self.batch_samples = ([],[])
        
    def run(self):
        while not self.is_max_stop:
            seed = self.find_initial()
            if seed is None: 
                if len(self.initial_samples)==0:
                    continue
                break
            self.chain_runs.append(ChainRun())
            is_stop, threshold = False, -np.inf
            while not is_stop:
                chain = self.seed_to_chain(seed, threshold)
                self.current_chain_run.add(chain)
                sample = chain.child_seed
                seed, threshold = sample.array, sample.performance
                is_stop = self.stop_chain_run(self.current_chain_run)
            self.update_reps_initials() 
    
    def find_initial(self):
        copy_sequence = self.attempt_sequence.copy()
        indicator = self.indicator_factory(-np.inf)
        while copy_sequence:
            initial_sample = self.sample(copy_sequence)
            if bool(indicator(initial_sample)):
                return initial_sample
        return None
    
    def sample_input(self,size):
        return norm.rvs(size=(size,self.dimension),random_state=self.random_state)

    def sample_noise(self,size):
        return norm.rvs(size=(size,self.dimension),random_state=self.random_state)

    def sample(self,sequence):
        if not self.batch_samples[0]:
            self.batch_samples = (list(self.sample_input(1000)),
                                  list(self.sample_noise(1000)))
        input = self.batch_samples[0].pop(0)
        noise = self.batch_samples[1].pop(0)
        scale = sequence.pop(0)
        return input + noise * scale
    
    def update_reps_initials(self):
        chain_run = self.current_chain_run
        fail_samples = chain_run.threshold_samples(self.threshold)
        if fail_samples:
            self.representatives.append(fail_samples[-1])
            self.initial_samples.append(fail_samples[-1])
        else:
            self.representatives.append(chain_run.current_chain.child_seed)

    def seed_to_chain(self, seed, threshold):
        chain_data = RecycleChainData(accept_chain_list=[[seed]],
                                      propose_chain_list=[[]])
        indicator = self.indicator_factory(threshold)
        self.markov_chain.update(chain_data,self.chain_len - 1,indicator)
        sample_list = [Sample(array=array,
                              performance=self.performance_function(array))
                              for array in chain_data.accept_chain_list[0]]
        return SingleChain(sample_list)
    
    def stop_chain_run(self, chain_run):
        curr_threshold = chain_run.current_chain.child_threshold
        cond_a = curr_threshold >= self.threshold
        cond_b = (sum([chain.child_threshold == curr_threshold
                      for chain in chain_run.chains]) >= self.converge_limit)
        cond_c = len(chain_run.chains) >= self.max_len
        return cond_a or cond_b or cond_c

    def hv_test(self,x,y):
        midpoint = (x.array + y.array) / 2
        mid_perf = self.performance_function(midpoint)
        return mid_perf >= min(x.performance,y.performance)
    
    def indicator_factory(self, threshold):
        def indicator(x):
            performance = self.performance_function(x)
            if performance >= threshold:
                sample = Sample(array=x,performance=performance)
                for rep in self.representatives:
                    if self.hv_test(sample,rep):
                        return 0
                return 1
            return 0
        return indicator
    
    @property
    def current_chain_run(self):
        return self.chain_runs[-1]
    
    @property
    def is_max_stop(self):
        n = len(self.initial_samples)
        print(f"\rInitial samples found: {n}", end="", flush=True)
        return n >= self.max_initial_samples
    
    def plot(self):
        for chain_run in self.chain_runs:
            chain_run.plot()