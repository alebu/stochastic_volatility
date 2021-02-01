import numpy as np

def simulate_random_walk(n_steps, probability_up = 0.5):
    binomial = np.random.binomial(1, p = probability_up, size = n_steps)
    return np.cumsum(
        np.where(
            binomial == 0,
            -1,
            binomial
        )
    )