"""
Entropic Optimal Transition Coupling (OTC) solvers.

Implements variants of the OTC algorithm using entropic regularization.
Includes both a custom Sinkhorn implementation and one based on the POT library.

References:
    - Section 5, "Optimal Transport for Stationary Markov Chains via Policy Iteration"
      (https://www.jmlr.org/papers/volume23/21-0519/21-0519.pdf)

Functions:
    - entropic_otc: Entropic OTC using a self-implemented Sinkhorn solver.
    - entropic_otc1: Entropic OTC using the POT library's Sinkhorn solver.
"""

import numpy as np
from ..utils import get_best_stat_dist
from .approx_tce import approx_tce
from .entropic_tci import entropic_tci, entropic_tci1


def entropic_otc(Px, Py, c, L=100, T=100, xi=0.1, sink_iter=100, get_sd=False):
    """
    Solves the Entropic Optimal Transition Coupling (OTC) problem between two Markov chains
    using approximate policy iteration and entropic regularization.

    This method alternates between approximate coupling evaluation
    and entropic coupling improvement (via Sinkhorn iterations), until convergence.

    Args:
        Px (np.ndarray): Transition matrix of the source Markov chain of shape (dx, dx).
        Py (np.ndarray): Transition matrix of the target Markov chain of shape (dy, dy).
        c (np.ndarray): Cost function of shape (dx, dy).
        L (int): Number of iterations for computing the cost vector g in approx_tce.
        T (int): Number of iterations for computing the bias vector h in approx_tce.
        xi (float): Scaling factor for entropic cost adjustment in entropic_tci.
        sink_iter (int): Number of Sinkhorn iterations in entropic_tci.
        get_sd (bool): If True, compute best stationary distribution using linear programming.

    Returns:
        exp_cost (float): Expected transport cost under the optimal transition coupling.
        P (np.ndarray): Optimal transition coupling matrix of shape (dx*dy, dx*dy).
        stat_dist (Optional[np.ndarray]): Stationary distribution of the optimal transition coupling of shape (dx, dy),
                                            or None if get_sd is False.
    """

    dx, dy = Px.shape[0], Py.shape[0]
    max_c = np.max(c)
    tol = 1e-5 * max_c

    g_old = max_c * np.ones(dx * dy)
    g = g_old - 10 * tol
    P = np.kron(Px, Py)
    iter_ctr = 0
    while g_old[0] - g[0] > tol:
        iter_ctr += 1
        P_old = P
        g_old = g

        # Approximate transition coupling evaluation
        g, h = approx_tce(P, c, L, T)

        # Entropic transition coupling improvement
        P = entropic_tci(h, P_old, Px, Py, xi, sink_iter)

    # In case of numerical instability, make non-negative and normalize.
    P = np.maximum(P, 0)
    row_sums = np.sum(P, axis=1, keepdims=True)
    P = P / np.where(row_sums > 0, row_sums, 1)

    if get_sd:
        stat_dist, exp_cost = get_best_stat_dist(P, c)
        stat_dist = np.reshape(stat_dist, (dx, dy))
    else:
        stat_dist = None
        exp_cost = g[0].item()

    return exp_cost, P, stat_dist


def entropic_otc1(
    Px, Py, c, L=100, T=100, xi=0.1, reg_num=0.1, sink_iter=100, get_sd=False
):
    dx, dy = Px.shape[0], Py.shape[0]
    max_c = np.max(c)
    tol = 1e-5 * max_c

    g_old = max_c * np.ones(dx * dy)
    g = g_old - 10 * tol
    P = np.kron(Px, Py)
    iter_ctr = 0
    while g_old[0] - g[0] > tol:
        iter_ctr += 1
        P_old = P
        g_old = g

        # Approximate transition coupling evaluation
        g, h = approx_tce(P, c, L, T)

        # Entropic transition coupling improvement
        P = entropic_tci1(h, P_old, Px, Py, xi, reg_num, sink_iter)

    # In case of numerical instability, make non-negative and normalize.
    P = np.maximum(P, 0)
    row_sums = np.sum(P, axis=1, keepdims=True)
    P = P / np.where(row_sums > 0, row_sums, 1)

    if get_sd:
        stat_dist, exp_cost = get_best_stat_dist(P, c)
        stat_dist = np.reshape(stat_dist, (dx, dy))
    else:
        stat_dist = None
        exp_cost = g[0].item()

    return exp_cost, P, stat_dist
