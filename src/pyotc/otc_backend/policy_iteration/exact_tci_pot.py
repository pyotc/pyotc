"""
Original Transition Coupling Improvements (TCI) methods from:
https://jmlr.csail.mit.edu/papers/volume23/21-0519/21-0519.pdf

Use the python optimal transport (POT) library to solve optimal transport problem.
"""

import numpy as np
import copy
from pyotc.otc_backend.optimal_transport.pot import computeot_pot

def check_constant(f, threshold=1e-3):  
    """
    Determines whether all elements of the vector f are approximately equal.

    Args:
        f (np.ndarray): Function vector of shape (d,).
        threshold (float): Tolerance for the maximum allowable difference between elements.

    Returns:
        bool: True if all elements of f differ by no more than the threshold; False otherwise.
    """
   
    d = f.shape[0]
    f_const = True
    for i in range(d):
        for j in range(i + 1, d):
            if abs(f[i] - f[j]) > threshold:
                f_const = False
                break
        if not f_const:
            break
    return f_const


def setup_ot(f, Px, Py, Pz):
    """
    This improvement step selects a new transition coupling matrix Pz that minimizes Pzf.
    In more detail, we may select a transition coupling Pz such that for each state pair (x, y), 
    the corresponding row r = Pz((x, y), ·) minimizes rf over couplings r in Pi(P(x, ·), Q(y, ·)).
    This is done by solving the optimal transport problem for each state pair (x, y) in the source
    and target Markov chains. The resulting transition coupling matrix Pz is updated accordingly.

    This function uses the POT (Python Optimal Transport) library to solve the optimal transport problem
    for each (x, y) state pair and updates the transition coupling matrix.

    Args:
        f (np.ndarray): Cost function reshaped as of shape (dx*dy,).
        Px (np.ndarray): Transition matrix of the source Markov chain of shape (dx, dx).
        Py (np.ndarray): Transition matrix of the target Markov chain of shape (dy, dy).
        Pz (np.ndarray): Transition coupling matrix to update of shape (dx*dy, dx*dy).

    Returns:
        np.ndarray: Updated transition coupling matrix Pz.
    """
    
    dx = Px.shape[0]
    dy = Py.shape[0]
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
                sol, val = computeot_pot(f_mat, dist_x, dist_y)
            idx = dy * (x_row) + y_row
            Pz[idx, :] = np.reshape(sol, (-1, dx * dy))
    return Pz


def exact_tci(g, h, P0, Px, Py):
    """
    Performs exact Transition Coupling Improvement (TCI) using optimal transport.

    This function iteratively refines a transition coupling matrix by solving OT problems
    with respect to coupling evaluation functions g and h. 
    
    Args:
        g (np.ndarray): TCE vector g of shape dx*dy.
        h (np.ndarray): TCE vector h of shape dx*dy.
        P0 (np.ndarray): Previous transition coupling matrix of shape (dx*dy, dx*dy).
        Px (np.ndarray): Transition matrix of the source Markov chain of shape (dx, dx).
        Py (np.ndarray): Transition matrix of the target Markov chain of shape (dy, dy).

    Returns:
        np.ndarray: Updated transition coupling matrix of shape (dx*dy, dx*dy).
    """
    
    # Check if g is constant.
    dx = Px.shape[0]
    dy = Py.shape[0]
    Pz = np.zeros((dx * dy, dx * dy))
    g_const = check_constant(f=g)
    
    # If g is not constant, improve transition coupling against g.
    if not g_const:
        Pz = setup_ot(f=g, Px=Px, Py=Py, Pz=Pz)
        if np.max(np.abs(np.matmul(P0, g) - np.matmul(Pz, g))) <= 1e-7:
            Pz = copy.deepcopy(P0)
        else:
            return Pz
        
    # Try to improve with respect to h.
    Pz = setup_ot(f=h, Px=Px, Py=Py, Pz=Pz)
    if np.max(np.abs(np.matmul(P0, h) - np.matmul(Pz, h))) <= 1e-4:
        Pz = copy.deepcopy(P0)
        
    return Pz
