import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.column_stack((np.ones(X.shape[0]),X))
    w = np.random.randn(X.shape[1])
    for step in range(steps):
        logits = X.dot(w)
        p = np.clip(_sigmoid(logits), 1e-7, 1-1e-7)
        grad =  (X.T.dot(p-y)) / len(X)
        w  = w - lr * grad
    return (w[1:], w[0])