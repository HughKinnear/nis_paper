import numpy as np
from scipy.stats import norm, uniform


class RecycleChainData:

    def __init__(self,
                 accept_chain_list,
                 propose_chain_list):
        self.accept_chain_list = accept_chain_list
        self.propose_chain_list = propose_chain_list

    @property
    def number_of_params(self):
        return len(self.accept_chain_list[0][0])
    
    @property
    def chain_number(self):
        return len(self.accept_chain_list)
    
    @property
    def all_accept_samples(self):
        return [item for chain in self.accept_chain_list for item in chain]
    
    @property
    def all_propose_samples(self):
        return [item for chain in self.propose_chain_list for item in chain]


class RecycleModifiedMetropolis:

    def __init__(self,
                 scale,
                 random_state):
        self.scale = scale
        self.random_state = random_state

    def update(self,recycle_chain_data,length,indicator):
        sample_shape = (recycle_chain_data.chain_number,
                        recycle_chain_data.number_of_params,
                        length)
        loguniform_samps = np.log(uniform.rvs(size=sample_shape,random_state=self.random_state))
        proposal_samps = norm.rvs(size=sample_shape,scale=self.scale,random_state=self.random_state)
        accept_state_list = [np.array([chain[-1] for chain in recycle_chain_data.accept_chain_list])]
        propose_state_list = []
        for i in range(length):
            current_state = accept_state_list[-1]
            prop_state = current_state + proposal_samps[:,:,i]
            alpha = norm.logpdf(prop_state) - norm.logpdf(current_state)
            accept = alpha >= loguniform_samps[:,:,i]
            new_state = np.where(accept, prop_state, current_state)
            propose_state_list.append(new_state)
            ind_accept = np.array([[bool(indicator(state))] for state in new_state])
            accept_state_list.append(np.where(ind_accept, new_state, current_state))
        for state in accept_state_list[1:]:
            for sample,chain in zip(state,recycle_chain_data.accept_chain_list):
                chain.append(sample)
        for state in propose_state_list:
            for sample,chain in zip(state,recycle_chain_data.propose_chain_list):
                chain.append(sample)


