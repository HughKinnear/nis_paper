import numpy as np
from scipy.stats import multivariate_normal
from dataclasses import dataclass


@dataclass
class ISResults:
    estimate: ...
    cross_entropy_weights: ...
    cov_estimate: ... 
    weight_cov_estimate: ...
    importance_values: ...
    indicators: ...
    importance_weights: ... 
    posterior_probs: ...


class ReliabilityImportanceSampler:

    def __init__(self,
                 importance_distribution,
                 performance_function,
                 threshold,
                 random_state):
        self.importance_distribution = importance_distribution
        self.performance_function = performance_function
        self.threshold = threshold
        self.random_state = random_state
        self.importance_samples = None

    def sample(self,budget):
        if self.importance_samples is None:
            self.importance_samples = self.importance_distribution.sample(budget)
        else:
            self.importance_samples = np.vstack((self.importance_samples,
                                                 self.importance_distribution.sample(budget)))

    
    def compute_stats(self):
        log_pdfs, posterior_probs = self.importance_distribution.logpdf(self.importance_samples)
        indicators = np.array([int(self.performance_function(samp) >= self.threshold)
                               for samp in self.importance_samples])
        dim = self.importance_distribution.dimension
        input_log_pdfs = multivariate_normal.logpdf(self.importance_samples,
                                                    mean=np.zeros(dim))
        importance_weights = np.exp(input_log_pdfs - log_pdfs)
        importance_values = indicators * importance_weights
        estimate = np.mean(importance_values)
        if estimate == 0:
            weight_cov_estimate = np.inf
            cov_estimate = np.inf
            n = posterior_probs.shape[1]
            cross_entropy_weights = np.ones(n) / n
        else:
            cross_entropy_weights = (sum(posterior_probs* importance_values[:,None])
                                    / sum(importance_values))
            weight_cov_estimate = np.std(importance_values) / estimate
            cov_estimate = weight_cov_estimate / np.sqrt(len(importance_values))
        results = ISResults(estimate=estimate,
                            cross_entropy_weights=cross_entropy_weights,
                            weight_cov_estimate=weight_cov_estimate,
                            cov_estimate=cov_estimate,
                            importance_values=importance_values,
                            indicators=indicators,
                            importance_weights=importance_weights,
                            posterior_probs=posterior_probs)
        return results
        

    
    
    
    


    

    


    


    








        
        
        
        


        



