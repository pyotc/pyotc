"""
Original Transition Coupling Evaluation (TCE) methods from:
https://jmlr.csail.mit.edu/papers/volume23/21-0519/21-0519.pdf
"""

import numpy as np
from numpy.linalg import pinv


# TODO: document, unit test, add warnings about pinv
def exact_tce(Pz, c):
    d = Pz.shape[0]
    c = np.reshape(c, (d, -1))
    A = np.block(
        [
            [np.eye(d) - Pz, np.zeros((d, d)), np.zeros((d, d))],
            [np.eye(d), np.eye(d) - Pz, np.zeros((d, d))],
            [np.zeros((d, d)), np.eye(d), np.eye(d) - Pz],
        ]
    )

    b = np.concatenate([np.zeros((d, 1)), c, np.zeros((d, 1))])
    try:
        sol = np.linalg.solve(A, b)
    except:
        sol = np.matmul(pinv(A), b)

    g = sol[0:d].flatten()
    h = sol[d : 2 * d].flatten()
    return g, h
