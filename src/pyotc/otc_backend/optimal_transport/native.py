"""
Native linear programming implementation for solving optimal transport (OT) problems.
"""

import numpy as np
from scipy.optimize import linprog


def computeot_lp(C, r, c):
    """
    Solves the optimal transport problem using linear programming (LP) with SciPy.

    Given a cost matrix `C` and distributions `r` and `c`, this function computes the
    optimal transport plan that minimizes the total transport cost.

    Args:
        C (np.ndarray): Cost matrix of shape (nx, ny), where C[i, j] represents the cost of transporting
                        mass from source i to target j.
        r (np.ndarray): Source distribution (shape: nx,). Should sum to 1.
        c (np.ndarray): Target distribution (shape: ny,). Should sum to 1.

    Returns:
        Tuple[np.ndarray, float]:
            - lp_sol (np.ndarray): Optimal transport plan of shape (nx, ny).
            - lp_val (float): Total transport cost under the optimal plan.
    """
    nx = r.size
    ny = c.size

    # setup LP
    Aeq = np.zeros((nx + ny, nx * ny))
    beq = np.concatenate((r.flatten(), c.flatten()))
    beq = beq.reshape(-1, 1)
    for row in range(nx):
        for t in range(ny):
            Aeq[row, (row * ny) + t] = 1
    for row in range(nx, nx + ny):
        for t in range(nx):
            Aeq[row, t * ny + (row - nx)] = 1
    cost = C.reshape(-1, 1)

    # Bound
    bound = [[0, None]] * (nx * ny)

    # Solve OT LP using linprog
    res = linprog(cost, A_eq=Aeq, b_eq=beq, bounds=bound, method="highs")
    lp_sol = res.x
    lp_val = res.fun
    return lp_sol, lp_val
