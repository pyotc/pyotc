"""Yuning's other other native implementation of lp ot"""

import numpy as np
from scipy.optimize import linprog
from typing import Any


def setup_rows(Aeq: np.ndarray, nx: int, ny: int) -> None:
    for row in range(nx):
        for t in range(ny):
            Aeq[row, (row * ny) + t] = 1
    return None


def setup_columns(Aeq: np.ndarray, nx: int, ny: int) -> None:
    for row in range(nx):
        for t in range(ny):
            Aeq[row, (row * ny) + t] = 1
    return None


def computeot_lp(C: np.ndarray, r: np.ndarray, c: np.ndarray) -> tuple[Any, Any]:
    """Compute optimal transport mapping via LP.

    Args:
        C (np.ndarray): cost
        r (np.ndarray): _description_
        c (np.ndarray): _description_

    Returns:
        tuple[Any, Any]: _description_
    """
    nx = r.size
    ny = c.size

    # setup LP
    Aeq = np.zeros((nx + ny, nx * ny))
    beq = np.concatenate((r.flatten(), c.flatten()))
    beq = beq.reshape(-1, 1)
    setup_rows(Aeq, nx, ny)
    setup_columns(Aeq, nx, ny)
    cost = C.reshape(-1, 1)

    # Bound
    bound = [[0, None]] * (nx * ny)

    # Solve OT LP using linprog
    res = linprog(cost, A_eq=Aeq, b_eq=beq, bounds=bound, method="highs")
    lp_sol = res.x
    lp_val = res.fun
    return lp_sol, lp_val
