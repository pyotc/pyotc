"""
Original Transition Coupling Improvements (TCI) methods from:
https://jmlr.csail.mit.edu/papers/volume23/21-0519/21-0519.pdf

Use scipy.linprog (LP solver) library to solve optimal transport problem.
"""

import numpy as np
import copy
from pyotc.otc_backend.optimal_transport.native import computeot_lp


def check_constant(f, Px, threshold=1e-3):
    dx = Px.shape[0]
    g_const = True
    for i in range(dx):
        for j in range(i + 1, dx):
            if abs(f[i] - f[j]) > threshold:
                g_const = False
                break
        if not g_const:
            break
    return g_const


def setup_ot(f, Px, Py, Pz):
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
                sol, val = computeot_lp(f_mat, dist_x, dist_y)
            idx = dy * (x_row) + y_row
            Pz[idx, :] = np.reshape(sol, (-1, dx * dy))
    return Pz


def exact_tci(g, h, P0, Px, Py):
    # Check if g is constant.
    dx = Px.shape[0]
    dy = Py.shape[0]
    Pz = np.zeros((dx * dy, dx * dy))
    g_const = check_constant(f=g, Px=Px)

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
