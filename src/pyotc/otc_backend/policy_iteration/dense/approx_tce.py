import numpy as np


def approx_tce(P, c, L, T):
    """
    Approximates the Transition Coupling Evaluation (TCE) vectors g and h
    using a truncation-based approximation of the exact TCE method.

    Args:
        P (np.ndarray): Transition matrix of shape (dx*dy, dx*dy).
        c (np.ndarray): Cost vector of shape (dx*dy,) or (dx*dy, 1).
        L (int): Maximum number of iterations for computing the cost vector g.
        T (int): Maximum number of iterations for computing the bias vector h.

    Returns:
        g (np.ndarray): Approximated average cost (gain) vector of shape (dx*dy,).
        h (np.ndarray): Approximated bias vector of shape (dx*dy,).
    """

    d = P.shape[0]
    c = np.reshape(c, (d, -1))
    c_max = np.max(c)

    g_old = c
    g = P @ g_old
    l = 1
    tol = 1e-12
    while l <= L and np.max(np.abs(g - g_old)) > tol * c_max:
        g_old = g
        g = P @ g_old
        l += 1

    g = np.mean(g) * np.ones((d, 1))
    diff = c - g
    h = diff.copy()
    t = 1
    while t <= T and np.max(np.abs(P @ diff)) > tol * c_max:
        h += P @ diff
        diff = P @ diff
        t += 1

    return g, h
