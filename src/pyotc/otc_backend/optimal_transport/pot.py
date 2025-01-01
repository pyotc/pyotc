"""A wrapper to python optimal transport POT"""

import numpy as np
import ot


def computeot_pot(C, r, c):
    # Ensure r and c are numpy arrays
    r = np.array(r).flatten()
    c = np.array(c).flatten()

    # Compute the optimal transport plan and the cost using the ot.emd function
    lp_sol = ot.emd(r, c, C)
    lp_val = np.sum(lp_sol * C)

    return lp_sol, lp_val
