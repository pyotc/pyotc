import numpy as np
import ot
from pyotc.otc_backend.optimal_transport.logsinkhorn import logsinkhorn


def entropic_tci(h, P0, Px, Py, xi, sink_iter):

    dx, dy = Px.shape[0], Py.shape[0]
    P = P0.copy()
    #P = copy.deepcopy(P0)
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

    dx, dy = Px.shape[0], Py.shape[0]
    P = P0.copy()
    #P = copy.deepcopy(P0)
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
