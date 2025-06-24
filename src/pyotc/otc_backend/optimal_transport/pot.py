"""
A wrapper for the Python Optimal Transport (POT) library.

This module provides a simplified interface for computing optimal transport plans
and their associated costs using the POT library.
"""

import numpy as np
import ot


def computeot_pot(C, r, c):
    """
    Computes the optimal transport plan and its total cost using the POT library.

    Given a cost matrix `C` and distributions `r` and `c`, this function computes the
    optimal transport plan that minimizes the total transport cost.

    Args:
        C (np.ndarray): Cost matrix of shape (n, m), where C[i, j] represents the cost of transporting
                        mass from source i to target j.
        r (np.ndarray): Source distribution (shape: n,). Should sum to 1.
        c (np.ndarray): Target distribution (shape: m,). Should sum to 1.

    Returns:
        Tuple[np.ndarray, float]:
            - lp_sol (np.ndarray): Optimal transport plan of shape (n, m).
            - lp_val (float): Total transport cost under the optimal plan.
    """
    # Ensure r and c are numpy arrays
    r = np.array(r).flatten()
    c = np.array(c).flatten()

    # Compute the optimal transport plan and the cost using the ot.emd function
    lp_sol = ot.emd(r, c, C)
    lp_val = np.sum(lp_sol * C)

    return lp_sol, lp_val
