import numpy as np
from examples.sis.ERADist import ERADist
from examples.sis.SIS_aCS import SIS_aCS
from dataclasses import dataclass
import warnings
from scipy.optimize import OptimizeWarning

@dataclass
class SISResult:
    failure_probability: ...
    levels: ...
    samples: ...
    samplesX: ...
    W_final: ...
    fs_iid: ...

def sis(d, g, N,seed,p=0.1, burn=0, tarCOV=1.5):
    warnings.filterwarnings("ignore", category=OptimizeWarning)
    np.random.seed(seed)
    samples_return= 2
    pi_pdf = [ERADist('standardnormal', 'PAR', np.nan) for _ in range(d)] 
    edit_g = lambda x: np.array([-g(x_i) for x_i in x])
    return SISResult(*SIS_aCS(N, p, edit_g, pi_pdf, burn, tarCOV,  samples_return))
