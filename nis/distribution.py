from scipy.stats import vonmises_fisher, multinomial
from scipy.special import logsumexp
from scipy.stats import nakagami
import numpy as np


class vMFNM:

    def __init__(self,
                 weights,
                 mus,
                 kappas,
                 omegas,
                 ms,
                 random_state):
        self.random_state = random_state
        self.weights = weights
        self.mus = mus
        self.kappas = kappas
        self.omegas = omegas
        self.ms = ms

    @property
    def n_components(self):
        return len(self.weights)
    
    @property
    def dimension(self):
        return self.mus.shape[1]
    
    @property
    def scipy_scales(self):
        return np.sqrt(self.omegas)
    
    
    def sample(self, num_samples):
        samples, _ = self.sample_with_comps(num_samples)
        return samples
    
    def sample_with_comps(self, num_samples):
        comps = multinomial.rvs(n=num_samples,p=self.weights,random_state=self.random_state)
        samples = []
        for i in range(self.n_components):
            directions = vonmises_fisher.rvs(mu=self.mus[i],
                                             kappa=self.kappas[i],
                                             size=comps[i],
                                             random_state=self.random_state)
            distances = nakagami.rvs(self.ms[i],
                                     scale=self.scipy_scales[i],
                                     size=comps[i],
                                     random_state=self.random_state)
            samples.append(directions * distances[:,None])
        samples = np.concatenate(samples)
        return samples,comps

    def logpdf(self,x):
        distances = np.linalg.norm(x,axis=1)
        directions = x / distances[:,None]
        direction_log_pdfs = np.array([vonmises_fisher.logpdf(directions,
                                                                mu=mu,
                                                                kappa=kappa)
                                        for mu,kappa in zip(self.mus,self.kappas)])
        distance_log_pdfs = np.array([nakagami.logpdf(distances,
                                                    m,
                                                    scale=scale)
                                    for m,scale in zip(self.ms,self.scipy_scales)])

        log_inv_jacobians = np.log(distances) * (1- self.dimension)
        a = direction_log_pdfs + distance_log_pdfs + log_inv_jacobians
        logpdfs = logsumexp(a,b=self.weights[:,None],axis=0)
        posterior_probs = (self.weights[:,None] * np.exp(a - logpdfs)).T
        return logpdfs, posterior_probs