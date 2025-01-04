"""Yuning's other other native implementation of lp ot"""

import numpy as np
from scipy.optimize import linprog


# TODO: Document & Unit test
def computeot_lp(C, r, c):
    # nx = r.shape[0]
    # ny = c.shape[1]
    nx = r.size
    ny = c.size
    Aeq = np.zeros((nx + ny, nx * ny))
    beq = np.concatenate((r.flatten(), c.flatten()))
    beq = beq.reshape(-1, 1)

    # column sums correct
    for row in range(nx):
        for t in range(ny):
            Aeq[row, (row * ny) + t] = 1

    # row sums correct
    for row in range(nx, nx + ny):
        for t in range(nx):
            Aeq[row, t * ny + (row - nx)] = 1

    # lb = np.zeros(nx*ny)
    bound = [[0, None]] * (nx * ny)

    # solve OT LP using linprog
    # cost = C.flatten()
    cost = C.reshape(-1, 1)
    # res = linprog(cost, A_eq=Aeq, b_eq=beq, bounds=(lb, None), method='highs')
    res = linprog(cost, A_eq=Aeq, b_eq=beq, bounds=bound, method="highs")
    lp_sol = res.x
    lp_val = res.fun
    return lp_sol, lp_val
