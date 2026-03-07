import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
        sigma_x = np.zeros_like(x, dtype=float)
    pos_x = x >=0
    ez_pos = np.exp(-1 * x[pos_x])
    ez_neg = np.exp(x[~pos_x])
    sigma_x[pos_x] = 1 / (1 + ez_pos)
    sigma_x[~pos_x] = ez_neg / (1 + ez_neg)
    return sigma_x