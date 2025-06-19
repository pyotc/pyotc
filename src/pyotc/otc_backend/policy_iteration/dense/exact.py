import numpy as np
from .exact_tce import exact_tce
from .exact_tci_lp import exact_tci as exact_tci_lp
from .exact_tci_pot import exact_tci as exact_tci_pot
from ..utils import get_ind_tc, get_stat_dist
import time

def exact_otc_lp(Px, Py, c, stat_dist='best'):
    start = time.time()
    print("Starting exact_otc_sparse...")
    
    dx = Px.shape[0]
    dy = Py.shape[0]
    P_old = np.ones((dx * dy, dx * dy))
    P = get_ind_tc(Px, Py)
    
    while np.max(np.abs(P - P_old)) > 1e-10:
        P_old = np.copy(P)

        print("Computing exact TCE...")
        g, h = exact_tce(P, c)

        print("Computing exact TCI...")
        P = exact_tci_lp(g, h, P_old, Px, Py)

        # Check for convergence.
        if np.all(P == P_old):
            print("Convergence reached. Computing stationary distribution...")
            stat_dist = get_stat_dist(P, method=stat_dist, c=c)
            stat_dist = np.reshape(stat_dist, (dx, dy))
            exp_cost = g[0].item()
            end = time.time()
            print(f"[exact_otc] Finished. Total time elapsed: {end - start:.3f} seconds.")
            return float(exp_cost), P, stat_dist

    return None, None, None


def exact_otc_pot(Px, Py, c, stat_dist='best'):
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
        Px (np.ndarray): Transition matrix of the source Markov chain of shape (dx, dx).
        Py (np.ndarray): Transition matrix of the target Markov chain of shape (dy, dy).
        c (np.ndarray): Cost function of shape (dx, dy).
        get_best_sd (bool): If True, compute the best stationary distribution and exact expected cost
                            via linear programming; otherwise, return any stationary distribution of the OTC.

    Returns:
        exp_cost (float): Expected transport cost under the optimal transition coupling.
        P (np.ndarray): Optimal transition coupling matrix of shape (dx*dy, dx*dy).
        stat_dist (np.ndarray): Stationary distribution of the optimal transition coupling of shape (dx, dy).

        Returns (None, None, None) if the algorithm fails to converge.
    """
    start = time.time()
    print("Starting exact_otc_sparse...")
    
    dx = Px.shape[0]
    dy = Py.shape[0]
    P_old = np.ones((dx * dy, dx * dy))
    P = get_ind_tc(Px, Py)
    
    while np.max(np.abs(P - P_old)) > 1e-10:
        P_old = np.copy(P)

        print("Computing exact TCE...")
        g, h = exact_tce(P, c)

        print("Computing exact TCI...")
        P = exact_tci_pot(g, h, P_old, Px, Py)

        # Check for convergence.
        if np.all(P == P_old):
            print("Convergence reached. Computing stationary distribution...")
            stat_dist = get_stat_dist(P, method=stat_dist, c=c)
            stat_dist = np.reshape(stat_dist, (dx, dy))
            exp_cost = g[0].item()
            end = time.time()
            print(f"[exact_otc] Finished. Total time elapsed: {end - start:.3f} seconds.")
            return float(exp_cost), P, stat_dist
        
    return None, None, None
