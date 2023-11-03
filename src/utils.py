import pandas as pd
import numpy as np
import scipy.stats as sp

def call_option_payoff(S, K):
    return pd.Series(S - K).clip(lower = 0)

def d_denominator(sigma, time_to_expiry):
    return sigma*np.sqrt(time_to_expiry)

def d_1(S, K, time_to_expiry, r, variance):
    d_1 = (
        np.log(S/K) + (
            time_to_expiry*(r + 0.5*variance)
        )
    )/d_denominator(np.sqrt(variance), time_to_expiry)
    return d_1

def d_2(S, K, time_to_expiry, r, variance):
    d_2 = (
        np.log(S/K) + (
            time_to_expiry*(r - 0.5*variance)
        )
    )/d_denominator(np.sqrt(variance), time_to_expiry)
    return d_2

def N(x):
    return sp.norm.cdf(x)
    
def N_prime(x):
    return np.exp(
        -((x**2)/2)
    )/np.sqrt(2*np.pi)

def call_option_bsm_formula(S, K, T, t, r, sigma):
    tau = T - t
    return S*N(d_1(S, K, tau, r, sigma**2)) - K*np.exp(-r*(tau))*N(d_2(S, K, tau, r, sigma**2))

def call_option_delta(S, K, T, t, r, sigma):
    # https://www.youtube.com/watch?v=54QFuJWYlOM
    # TODO This is missing interest rates!
    return N(d_1(S, K, T - t, r, sigma**2))

def call_option_gamma(S, K, T, t, r, sigma):
    return N_prime(d_1(S, K, T - t, r, sigma**2))/(S*sigma*np.sqrt(T - t))

def realised_vol(S):
    # see Shreve pag 106
    return (
        (   
            np.log(
                pd.Series(S).shift()/pd.Series(S)
            )           
        )**2
    ).sum()

def historical_vol(S, T):
    # see Shreve pag 106
    # this is only one possible way of estimating historical vol. For other estimators
    # see Wilmott page 815
    return np.sqrt(realised_vol(S)/(T))



def resample_underlying_simulation(S, delta_t):
    # resamples data for discrete hedging
    # delta_t is the ratio between simulation 
    # and hedging frequency. E.g., if S is ]
    # daily stock price but hedging happens every
    # two days, delta_t is 2
    n_simulation_steps = len(S)
    n_resampling_steps = int(n_simulation_steps/delta_t)
    resampling_steps = np.array([int(delta_t*i) for i in range(n_resampling_steps)])
    S_resampled = S[resampling_steps].copy()
    t = np.arange(n_simulation_steps)/(n_simulation_steps - 1)
    t_resampled = t[resampling_steps].copy()
    return t, S_resampled, t_resampled