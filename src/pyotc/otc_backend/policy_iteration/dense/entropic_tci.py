import numpy as np
import ot


def entropic_tci(h, P0, Px, Py, xi, solver_fn):
    """
    Performs entropic Transition Coupling Improvement (TCI) using log-domain Sinkhorn algorithm.

    For each (i, j) state pair from the product space of two Markov chains, this function solves
    a local entropic optimal transport problem based on the bias vector h.

    Args:
        h (np.ndarray): Bias vector of shape (dx*dy,).
        P0 (np.ndarray): Previous transition coupling matrix of shape (dx*dy, dx*dy).
        Px (np.ndarray): Transition matrix of the source Markov chain of shape (dx, dx).
        Py (np.ndarray): Transition matrix of the target Markov chain of shape (dy, dy).
        xi (float): Scaling factor for entropic cost adjustment.
        solver_fn (callable): A function solves the optimization and provides a transport plan. Specified in 'entropic_otc'.

    Returns:
        np.ndarray: Updated transition coupling matrix of shape (dx*dy, dx*dy).
    """

    dx, dy = Px.shape[0], Py.shape[0]
    P = P0.copy()
    h_mat = np.reshape(h, (dx, dy))
    K = -xi * h_mat

    for i in range(dx):
        for j in range(dy):
            dist_x = Px[i, :]
            dist_y = Py[j, :]
            x_idxs = np.where(dist_x > 0)[0]
            y_idxs = np.where(dist_y > 0)[0]

            if len(x_idxs) == 1 or len(y_idxs) == 1:
                P[dy * i + j, :] = P0[dy * i + j, :]
            else:
                A_matrix = K[np.ix_(x_idxs, y_idxs)]
                sub_dist_x = dist_x[x_idxs]
                sub_dist_y = dist_y[y_idxs]

                sol = solver_fn(A_matrix, sub_dist_x, sub_dist_y)

                sol_full = np.zeros((dx, dy))
                sol_full[np.ix_(x_idxs, y_idxs)] = sol
                P[dy * i + j, :] = sol_full.flatten()

    return P
