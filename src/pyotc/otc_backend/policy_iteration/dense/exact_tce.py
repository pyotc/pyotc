"""
Original Transition Coupling Evaluation (TCE) methods from:
https://www.jmlr.org/papers/volume23/21-0519/21-0519.pdf
"""

import numpy as np
from numpy.linalg import pinv


def exact_tce(R, c):
    """
    Computes the exact Transition Coupling Evaluation (TCE) vectors g and h
    using the linear system described in Algorithm 1a of the paper
    "Optimal Transport for Stationary Markov Chains via Policy Iteration"
    (https://www.jmlr.org/papers/volume23/21-0519/21-0519.pdf).

    The method solves a block linear system involving the transition matrix R and cost vector c.
    If the system is not full rank, a pseudo-inverse (pinv) is used as fallback.

    Args:
        R (np.ndarray): Transition matrix of shape (dx*dy, dx*dy).
        c (np.ndarray): Cost vector of shape (dx*dy, dx*dy).

    Returns:
        g (np.ndarray): Average cost (gain) vector of shape (dx*dy,).
        h (np.ndarray): Total extra cost (bias) vector of shape (dx*dy,).

    Notes:
        - If the matrix A is singular or ill-conditioned, the solution uses `np.linalg.pinv`,
          which may lead to numerical instability.
        - Make sure Pz is a proper stochastic matrix (rows sum to 1).
    """
    d = R.shape[0]
    c = np.reshape(c, (d, -1))

    # Construct the block matrix A and right-hand side vector b
    A = np.block(
        [
            [np.eye(d) - R, np.zeros((d, d)), np.zeros((d, d))],
            [np.eye(d), np.eye(d) - R, np.zeros((d, d))],
            [np.zeros((d, d)), np.eye(d), np.eye(d) - R],
        ]
    )
    b = np.concatenate([np.zeros((d, 1)), c, np.zeros((d, 1))])

    # Solve the linear system Ax = b
    try:
        sol = np.linalg.solve(A, b)
    except:
        sol = np.matmul(pinv(A), b)

    # Extract g and h from the solution
    g = sol[0:d].flatten()
    h = sol[d : 2 * d].flatten()

    return g, h
