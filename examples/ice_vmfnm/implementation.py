from examples.ice_vmfnm.ERADist import ERADist
from examples.ice_vmfnm.iCE_vMFNM import iCE_vMFNM
import numpy as np
from dataclasses import dataclass
import warnings
from scipy.optimize import OptimizeWarning


@dataclass
class ICEResult:
    failure_probability: ...
    levels: ...
    performance_evals: ...
    samples: ...
    samplesX: ...
    k_fin: ...
    W_final: ...
    fs_iid: ...


def ice_vmfnm(d, g, max_it, N, k_init, seed, CV_target=1.5):
    warnings.filterwarnings("ignore", category=OptimizeWarning)
    samples_return= 2
    p = 0.1
    random_state = np.random.default_rng(seed)
    pi_pdf = [
        ERADist("standardnormal", "PAR", [0, 1],random_state=random_state) for i in range(d)
    ]
    edit_g = lambda x: np.array([-g(x_i) for x_i in x])
    return ICEResult(*iCE_vMFNM(N, p, edit_g, pi_pdf, max_it, CV_target, k_init, samples_return, random_state))




