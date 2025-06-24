"""
Original Transition Coupling Improvements (TCI) method from:
https://www.jmlr.org/papers/volume23/21-0519/21-0519.pdf

Use the python optimal transport (POT) library to solve optimal transport problem.
"""

import numpy as np
import copy
from pyotc.otc_backend.optimal_transport.pot import computeot_pot


def setup_ot(f, Px, Py, R):
    """
    This improvement step updates the transition coupling matrix R that minimizes the product Rf element-wise.
    In more detail, we may select a transition coupling R such that for each state pair (x, y),
    the corresponding row r = R((x, y), ·) minimizes rf over couplings r in Pi(Px(x, ·), Py(y, ·)).
    This is done by solving the optimal transport problem for each state pair (x, y) in the source
    and target Markov chains. The resulting transition coupling matrix R is updated accordingly.

    This function uses the POT (Python Optimal Transport) library to solve the optimal transport problem
    for each (x, y) state pair and updates the transition coupling matrix.

    Args:
        f (np.ndarray): Cost function reshaped as of shape (dx*dy,).
        Px (np.ndarray): Transition matrix of the source Markov chain of shape (dx, dx).
        Py (np.ndarray): Transition matrix of the target Markov chain of shape (dy, dy).
        R (np.ndarray): Transition coupling matrix to update of shape (dx*dy, dx*dy).

    Returns:
        R (np.ndarray): Updated transition coupling matrix of shape (dx*dy, dx*dy).
    """

    dx, dy = Px.shape[0], Py.shape[0]
    f_mat = np.reshape(f, (dx, dy))

    for x_row in range(dx):
        for y_row in range(dy):
            dist_x = Px[x_row, :]
            dist_y = Py[y_row, :]

            # Check if either distribution is degenerate.
            if any(dist_x == 1) or any(dist_y == 1):
                sol = np.outer(dist_x, dist_y)
            # If not degenerate, proceed with OT.
            else:
                sol, _ = computeot_pot(f_mat, dist_x, dist_y)
            idx = dy * (x_row) + y_row
            R[idx, :] = np.reshape(sol, (-1, dx * dy))

    return R


def exact_tci(g, h, R0, Px, Py):
    """
    Performs the Transition Coupling Improvement (TCI) step in the OTC algorithm.

    This function attempts to update the current coupling transition matrix R0
    based on the evaluation vectors g and h obtained from the Transition Coupling Evaluation (TCE).

    Args:
        g (np.ndarray): Gain vector from TCE of shape (dx*dy,).
        h (np.ndarray): Bias vector from TCE of shape (dx*dy,).
        R0 (np.ndarray): Current transition coupling matrix of shape (dx*dy, dx*dy).
        Px (np.ndarray): Transition matrix of the source Markov chain of shape (dx, dx).
        Py (np.ndarray): Transition matrix of the target Markov chain of shape (dy, dy).

    Returns:
        R (np.ndarray): Improved transition coupling matrix of shape (dx*dy, dx*dy).
    """

    # Check if g is constant.
    dx, dy = Px.shape[0], Py.shape[0]
    R = np.zeros((dx * dy, dx * dy))
    g_const = np.max(g) - np.min(g) <= 1e-3

    # If g is not constant, improve transition coupling against g.
    if not g_const:
        R = setup_ot(g, Px, Py, R)
        if np.max(np.abs(np.matmul(R0, g) - np.matmul(R, g))) <= 1e-7:
            R = copy.deepcopy(R0)
        else:
            return R

    # Try to improve with respect to h.
    R = setup_ot(h, Px, Py, R)
    if np.max(np.abs(np.matmul(R0, h) - np.matmul(R, h))) <= 1e-4:
        R = copy.deepcopy(R0)

    return R
