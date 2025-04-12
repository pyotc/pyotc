import numpy as np
from .exact_tce import exact_tce
from .exact_tci_lp import exact_tci as exact_tci_lp
from .exact_tci_pot import exact_tci as exact_tci_pot
from .utils import get_ind_tc, get_stat_dist, get_best_stat_dist


def exact_otc_lp(Px, Py, c, get_best_sd=True):
    dx = Px.shape[0]
    dy = Py.shape[0]

    P_old = np.ones((dx * dy, dx * dy))
    P = get_ind_tc(Px, Py)
    while np.max(np.abs(P - P_old)) > 1e-10:
        P_old = np.copy(P)

        # Transition coupling evaluation.
        g, h = exact_tce(P, c)

        # Transition coupling improvement.
        P = exact_tci_lp(g, h, P_old, Px, Py)

        # Check for convergence.
        if np.all(P == P_old):
            if get_best_sd:
                stat_dist, exp_cost = get_best_stat_dist(P, c)
                stat_dist = np.reshape(stat_dist, (dx, dy))                
            else:         
                stat_dist = get_stat_dist(P)
                stat_dist = np.reshape(stat_dist, (dx, dy))
                exp_cost = g[0].item()
            return float(exp_cost), P, stat_dist

    return None, None, None


def exact_otc_pot(Px, Py, c, get_best_sd=True):
    """
    Solves the Optimal Transition Coupling (OTC) problem between two Markov chains 
    using policy iteration, as described in Algorithm 1 of the paper:
    "Optimal Transport for Stationary Markov Chains via Policy Iteration" 
    (https://www.jmlr.org/papers/volume23/21-0519/21-0519.pdf).

    The algorithm iteratively updates the transition coupling matrix until convergence
    by alternating between Transition Coupling Evaluation (TCE) and Transition Coupling 
    Improvement (TCI) steps.
    
    For a detailed discussion of the connection between the OTC problem and Markov Decision Processes (MDPs), see Section 4 of the paper.
    Additional background on policy iteration methods for solving average-cost MDP problems 
    can be found in Chapters 8 and 9 of "Markov Decision Processes: Discrete Stochastic Dynamic Programming" by Martin L. Puterman.

    Args:
        Px (np.ndarray): Transition matrix of the source Markov chain (shape: dx * dx).
        Py (np.ndarray): Transition matrix of the target Markov chain (shape: dy * dy).
        c (np.ndarray): Cost function (shape: dx * dy).
        get_best_sd (bool): If True, compute the best stationary distribution and exact expected cost
                            via linear programming; otherwise, return any stationary distribution of the OTC.

    Returns:
        exp_cost (float): Expected transport cost under the optimal transition coupling.
        P (np.ndarray): Optimal transition coupling matrix (shape: dx*dy*dx*dy).
        stat_dist (np.ndarray): Stationary distribution of the optimal transition coupling (shape: dx*dy).

        Returns (None, None, None) if the algorithm fails to converge.
    """
    
    dx = Px.shape[0]
    dy = Py.shape[0]

    P_old = np.ones((dx * dy, dx * dy))
    P = get_ind_tc(Px, Py)
    while np.max(np.abs(P - P_old)) > 1e-10:
        P_old = np.copy(P)

        # Transition coupling evaluation.
        g, h = exact_tce(P, c)

        # Transition coupling improvement.
        P = exact_tci_pot(g, h, P_old, Px, Py)

        # Check for convergence.
        if np.all(P == P_old):
            if get_best_sd:
                stat_dist, exp_cost = get_best_stat_dist(P, c)
                stat_dist = np.reshape(stat_dist, (dx, dy))                
            else:         
                stat_dist = get_stat_dist(P)
                stat_dist = np.reshape(stat_dist, (dx, dy))
                exp_cost = g[0].item()
            return float(exp_cost), P, stat_dist

    return None, None, None
