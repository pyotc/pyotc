import numpy as np
import ot
from pyotc.otc_backend.optimal_transport.logsinkhorn import logsinkhorn


def entropic_tci(h, P0, Px, Py, xi, sink_iter):
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
        sink_iter (int): Number of iterations for the log-Sinkhorn solver.

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
                sol = logsinkhorn(A_matrix, sub_dist_x, sub_dist_y, sink_iter)
                sol_full = np.zeros((dx, dy))
                sol_full[np.ix_(x_idxs, y_idxs)] = sol
                P[dy * i + j, :] = sol_full.flatten()

    return P


def entropic_tci1(h, P0, Px, Py, xi, reg_num, sink_iter):
    """
    Performs entropic Transition Coupling Improvement (TCI) using the Sinkhorn algorithm from POT.

    Args:
        h (np.ndarray): Bias vector of shape (dx*dy,).
        P0 (np.ndarray): Previous transition coupling matrix of shape (dx*dy, dx*dy).
        Px (np.ndarray): Transition matrix of the source Markov chain of shape (dx, dx).
        Py (np.ndarray): Transition matrix of the target Markov chain of shape (dy, dy).
        xi (float): Scaling factor for entropic cost adjustment.
        reg_num (float): Regularization strength for the Sinkhorn solver.
        sink_iter (int): Maximum number of Sinkhorn iterations.

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
                sol = ot.sinkhorn(sub_dist_x, sub_dist_y, A_matrix, reg = reg_num, numItermax = sink_iter)
                #sol = logsinkhorn(A_matrix, sub_dist_x, sub_dist_y, sink_iter)
                sol_full = np.zeros((dx, dy))
                sol_full[np.ix_(x_idxs, y_idxs)] = sol
                P[dy * i + j, :] = sol_full.flatten()

    return P
