"""
Original Transition Coupling Evaluation (TCE) methods from:
https://www.jmlr.org/papers/volume23/21-0519/21-0519.pdf 
"""

import numpy as np
from numpy.linalg import pinv


# TODO: document, unit test, add warnings about pinv
def exact_tce(Pz, c):
    """
    Computes the exact Transition Coupling Evaluation (TCE) vectors g and h 
    using the linear system described in Algorithm 1a of the paper 
    "Optimal Transport for Stationary Markov Chains via Policy Iteration"
    (https://www.jmlr.org/papers/volume23/21-0519/21-0519.pdf).

    The method solves a block linear system involving the transition matrix Pz and cost vector c.
    If the system is not full rank, a pseudo-inverse (pinv) is used as fallback.

    Args:
        Pz (np.ndarray): Transition matrix of shape (d, d).
        c (np.ndarray): Cost vector of shape (d,) or (d, 1).

    Returns:
        g (np.ndarray): First coupling vector of shape (d,).
        h (np.ndarray): Second coupling vector of shape (d,).

    Notes:
        - If the matrix A is singular or ill-conditioned, the solution uses `np.linalg.pinv`, 
          which may lead to numerical instability.
        - Make sure Pz is a proper stochastic matrix (rows sum to 1).
    """
    d = Pz.shape[0]
    c = np.reshape(c, (d, -1))
    A = np.block(
        [
            [np.eye(d) - Pz, np.zeros((d, d)), np.zeros((d, d))],
            [np.eye(d), np.eye(d) - Pz, np.zeros((d, d))],
            [np.zeros((d, d)), np.eye(d), np.eye(d) - Pz],
        ]
    )

    b = np.concatenate([np.zeros((d, 1)), c, np.zeros((d, 1))])
    try:
        sol = np.linalg.solve(A, b)
    except:
        sol = np.matmul(pinv(A), b)

    g = sol[0:d].flatten()
    h = sol[d:2*d].flatten()
    return g, h
