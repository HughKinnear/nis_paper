import numpy as np
from scipy.stats import norm, gamma
from scipy.linalg import eigh

def covnert_performance_function(perf_function,orig_dim):

    def new_performance_function(x):
        new_dim_multiplier = len(x)//orig_dim
        new_input = [sum(x[new_dim_multiplier*i:new_dim_multiplier*(i+1)])
                     / np.sqrt(new_dim_multiplier)
                     for i in range(orig_dim)]
        return perf_function(np.array(new_input))
    
    return new_performance_function

def convert_array(x):
    half_dim = len(x) // 2
    x_1 = sum(x[:half_dim]) / np.sqrt(half_dim)
    x_2 = sum(x[half_dim:]) / np.sqrt(half_dim)
    return np.array([x_1,x_2])

def pwl(x):
    if x[0] > 3.5:
        g_1 = 4 - x[0]
    else:
        g_1 = 0.85 - (0.1 * x[0])
    if x[1] > 2:
        g_2 = 0.5 - (0.1 * x[1])
    else:
        g_2 = 2.3 - x[1]
    return -min([g_1, g_2])

c_pwl = covnert_performance_function(pwl,2)

def meatball(x):
    x_1 = x[0]
    x_2 = x[1]
    denom_a = 4*(x_1+2)**2/9
    denom_b = x_2**2/25
    denom_c = (x_1-2.5)**2/4
    denom_d = (x_2-0.5)**2/25
    return -(30/((denom_a + denom_b)**2 +1) + 20/((denom_c + denom_d)**2 +1) -5)

c_meatball = covnert_performance_function(meatball,2)

def two_dof(x):
    
    m1 = 2000
    m2 = 2000
    M = np.diag([m1, m2])
    sigma = np.sqrt(np.log(1.04))
    mu = np.log(2.5*10**5) - 0.5 * sigma**2
    x = np.array(x)
    k = np.exp(mu + sigma * x)
    k1 = k[0]
    k2 = k[1]
    K = np.array([[k1 + k2, -k2], [-k2, k2]])
    OME = 11
    tm = np.arange(0, 20.01, 0.01)
    
    evals, evecs = eigh(K, M)
    nat_freq = np.sqrt(evals)
    sort_ind = np.argsort(nat_freq)
    nat_freq = nat_freq[sort_ind]
    evecs = evecs[:, sort_ind]

    mass_norm_evec = evecs / np.sqrt(np.diag(evecs.T @ M @ evecs))

    mod_resp = np.full((len(tm), 2), np.nan)
    
    for i in range(2):
        w = nat_freq[i]
        r = OME / w
        eta = 0.02
        tanth = 2 * eta * r / (1 - r**2)
        sinth = np.sqrt(1- 1 / (1 + tanth**2))
        costh = np.sign(tanth) * np.sqrt(1 / (1 + tanth**2))
        wd = w * np.sqrt(1 - eta**2)
        A2 = sinth * (2000 * mass_norm_evec[1, i]) / (w**2 * np.sqrt((1 - r**2)**2 + (2 * eta * r)**2))
        A1 = 1 / wd * (A2 * eta * w - (2000 * mass_norm_evec[1, i]) * OME * costh / 
                       (w**2 * np.sqrt((1 - r**2)**2 + (2 * eta * r)**2)))
        p = np.exp(-eta * w * tm) * (A1 * np.sin(wd * tm) + A2 * np.cos(wd * tm)) + \
            (2000 * mass_norm_evec[1, i]) * (np.sin(OME * tm) * costh - np.cos(OME * tm) * sinth) / \
            (w**2 * np.sqrt((1 - r**2)**2 + (2 * eta * r)**2))
        mod_resp[:, i] = np.real(p)
    mass_resp = mass_norm_evec @ mod_resp.T
    y = np.max(mass_resp[0, :]) - 0.024
    return y

c_two_dof = covnert_performance_function(two_dof,2)

def suspension(x):
    c = x[0] * 10 + 424
    c_k = x[1] * 10 + 1480
    k = x[2] * 10 + 47
    A = 1
    b_0 = 0.27
    V = 1000
    M = 3.2633
    G = 981
    m = 0.8158
    term_1 = (np.pi*A*V*m)/(b_0*(k*G**2))
    term_2 = (((c_k/(M+m))-(c/m))**(2)) + ((c**2)/(M*m)) + (c_k*(k**2))/(m*(M**2))
    return -((term_1 * term_2) - 1)

c_suspension = covnert_performance_function(suspension,3)

def portfolio_low(x):
    b = 0.45
    q = 0.25
    eps = 1e-9
    x = np.array(x)
    U = x[0]
    mu = x[1]
    eta = x[2:]
    term_1 = (q*U + 3*(1-q**2)**(1/2)*eta)
    norm_cdf_mu = norm.cdf(mu)
    inv_gamma_cdf = gamma.ppf(norm_cdf_mu, a=6, scale=1/6)
    term_2 = inv_gamma_cdf ** (-1/2)
    n = len(eta)
    return np.sum((term_1 * term_2) >= 0.5*np.sqrt(n)) - b*n - eps


def portfolio_high(x):
    b = 0.25
    q = 0.25
    eps = 1e-9
    x = np.array(x)
    U = x[0]
    mu = x[1]
    eta = x[2:]
    term_1 = (q*U + 3*(1-q**2)**(1/2)*eta)
    norm_cdf_mu = norm.cdf(mu)
    inv_gamma_cdf = gamma.ppf(norm_cdf_mu, a=6, scale=1/6)
    term_2 = inv_gamma_cdf ** (-1/2)
    n = len(eta)
    return np.sum((term_1 * term_2) >= 0.5*np.sqrt(n)) - b*n - eps

    