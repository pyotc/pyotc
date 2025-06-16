import numpy as np
from scipy.optimize import linprog
import ot


def round_transpoly(X, r, c):
    A = X.copy()
    # A = copy.deepcopy(X)
    n1, n2 = A.shape

    r_A = np.sum(A, axis=1)
    for i in range(n1):
        scaling = min(1, r[i] / r_A[i])
        A[i, :] *= scaling

    c_A = np.sum(A, axis=0)
    for j in range(n2):
        scaling = min(1, c[j] / c_A[j])
        A[:, j] *= scaling

    r_A = np.sum(A, axis=1)
    c_A = np.sum(A, axis=0)
    err_r = r_A - r
    err_c = c_A - c

    if not np.all(err_r == 0) and not np.all(err_c == 0):
        A += np.outer(err_r, err_c) / np.sum(np.abs(err_r))

    return A


def frobinnerproduct(A, B):
    return np.sum(A * B)


def logsumexp(X, axis=None):
    y = np.max(
        X, axis=axis, keepdims=True
    )  # use 'keepdims' to make matrix operation X-y work
    s = y + np.log(np.sum(np.exp(X - y), axis=axis, keepdims=True))

    return np.squeeze(s, axis=axis)


def logsinkhorn(A, r, c, T):
    dx, dy = A.shape
    f = np.zeros(dx)
    g = np.zeros(dy)

    for t in range(T):
        if t % 2 == 0:
            f = np.log(r) - logsumexp(A + g, axis=1)
        else:
            g = np.log(c) - logsumexp(A + f[:, np.newaxis], axis=0)

    P = round_transpoly(np.exp(f[:, np.newaxis] + A + g), r, c)

    return P


def approx_tce(P, c, L, T):
    d = P.shape[0]
    c = np.reshape(c, (d, -1))
    c_max = np.max(c)

    g_old = c
    g = P @ g_old
    l = 1
    tol = 1e-12
    while l <= L and np.max(np.abs(g - g_old)) > tol * c_max:
        g_old = g
        g = P @ g_old
        l += 1

    g = np.mean(g) * np.ones((d, 1))
    diff = c - g
    h = diff.copy()
    t = 1
    while t <= T and np.max(np.abs(P @ diff)) > tol * c_max:
        h += P @ diff
        diff = P @ diff
        t += 1

    return g, h


def entropic_tci(h, P0, Px, Py, xi, sink_iter):
    dx, dy = Px.shape[0], Py.shape[0]
    P = P0.copy()
    # P = copy.deepcopy(P0)
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
    # P = copy.deepcopy(P0)
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
                sol = ot.sinkhorn(
                    sub_dist_x, sub_dist_y, A_matrix, reg=reg_num, numItermax=sink_iter
                )
                # sol = logsinkhorn(A_matrix, sub_dist_x, sub_dist_y, sink_iter)
                sol_full = np.zeros((dx, dy))
                sol_full[np.ix_(x_idxs, y_idxs)] = sol
                P[dy * i + j, :] = sol_full.flatten()

    return P


# def entropic_tci2(h, P0, Px, Py, xi, reg_num, sink_iter):

#     dx, dy = Px.shape[0], Py.shape[0]
#     P = P0.copy()
#     #P = copy.deepcopy(P0)
#     h_mat = np.reshape(h, (dx, dy))
#     K = -xi * h_mat
#     solver = Sinkhorn()

#     for i in range(dx):
#         for j in range(dy):
#             dist_x = Px[i, :]
#             dist_y = Py[j, :]
#             x_idxs = np.where(dist_x > 0)[0]
#             y_idxs = np.where(dist_y > 0)[0]

#             if len(x_idxs) == 1 or len(y_idxs) == 1:
#                 P[dy * i + j, :] = P0[dy * i + j, :]
#             else:
#                 A_matrix = K[np.ix_(x_idxs, y_idxs)]
#                 sub_dist_x = dist_x[x_idxs]
#                 sub_dist_y = dist_y[y_idxs]
#                 sol = solver(sub_dist_x, sub_dist_y, A_matrix)
#                 sol_full = np.zeros((dx, dy))
#                 sol_full[np.ix_(x_idxs, y_idxs)] = sol
#                 P[dy * i + j, :] = sol_full.flatten()

#     return P


def get_ind_tc(Px, Py):
    dx, dx_col = Px.shape
    dy, dy_col = Py.shape

    P_ind = np.zeros((dx * dy, dx_col * dy_col))
    for x_row in range(dx):
        for x_col in range(dx_col):
            for y_row in range(dy):
                for y_col in range(dy_col):
                    idx1 = dy * (x_row) + y_row
                    idx2 = dy * (x_col) + y_col
                    P_ind[idx1, idx2] = Px[x_row, x_col] * Py[y_row, y_col]
    return P_ind


def get_best_stat_dist(P, c):
    # Set up constraints.
    n = P.shape[0]
    c = np.reshape(c, (n, -1))
    Aeq = np.concatenate((P.T - np.eye(n), np.ones((1, n))), axis=0)
    beq = np.concatenate((np.zeros((n, 1)), 1), axis=None)
    beq = beq.reshape(-1, 1)
    bound = [[0, None]] * n

    # Solve linear program.
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bound)
    stat_dist = res.x
    exp_cost = res.fun

    return stat_dist, exp_cost


def entropic_otc(Px, Py, c, L=100, T=100, xi=0.1, sink_iter=10, get_sd=False):
    dx, dy = Px.shape[0], Py.shape[0]
    max_c = np.max(c)
    tol = 1e-5 * max_c

    g_old = max_c * np.ones(dx * dy)
    g = g_old - 10 * tol
    P = get_ind_tc(Px, Py)
    iter_ctr = 0
    while g_old[0] - g[0] > tol:
        iter_ctr += 1
        P_old = P
        g_old = g

        # Approximate transition coupling evaluation
        g, h = approx_tce(P, c, L, T)

        # Entropic transition coupling improvement
        P = entropic_tci(h, P_old, Px, Py, xi, sink_iter)

    # In case of numerical instability, make non-negative and normalize.
    P = np.maximum(P, 0)
    row_sums = np.sum(P, axis=1, keepdims=True)
    P = P / np.where(row_sums > 0, row_sums, 1)

    if get_sd:
        stat_dist, exp_cost = get_best_stat_dist(P, c)
        stat_dist = np.reshape(stat_dist, (dx, dy))
    else:
        stat_dist = None
        exp_cost = g[0].item()

    return exp_cost, P, stat_dist


def entropic_otc1(
    Px, Py, c, L=100, T=100, xi=0.1, reg_num=0.1, sink_iter=10, get_sd=False
):
    dx, dy = Px.shape[0], Py.shape[0]
    max_c = np.max(c)
    tol = 1e-5 * max_c

    g_old = max_c * np.ones(dx * dy)
    g = g_old - 10 * tol
    P = get_ind_tc(Px, Py)
    iter_ctr = 0
    while g_old[0] - g[0] > tol:
        iter_ctr += 1
        P_old = P
        g_old = g

        # Approximate transition coupling evaluation
        g, h = approx_tce(P, c, L, T)

        # Entropic transition coupling improvement
        P = entropic_tci1(h, P_old, Px, Py, xi, reg_num, sink_iter)

    # In case of numerical instability, make non-negative and normalize.
    P = np.maximum(P, 0)
    row_sums = np.sum(P, axis=1, keepdims=True)
    P = P / np.where(row_sums > 0, row_sums, 1)

    if get_sd:
        stat_dist, exp_cost = get_best_stat_dist(P, c)
        stat_dist = np.reshape(stat_dist, (dx, dy))
    else:
        stat_dist = None
        exp_cost = g[0].item()

    return exp_cost, P, stat_dist


# def entropic_otc2(Px, Py, c, L = 100, T = 100, xi = 0.1, reg_num = 0.1, sink_iter = 10, get_sd = False):

#     dx, dy = Px.shape[0], Py.shape[0]
#     max_c = np.max(c)
#     tol = 1e-5 * max_c

#     g_old = max_c * np.ones(dx * dy)
#     g = g_old - 10 * tol
#     P = get_ind_tc(Px, Py)
#     iter_ctr = 0
#     while g_old[0] - g[0] > tol:
#         iter_ctr += 1
#         P_old = P
#         g_old = g

#         # Approximate transition coupling evaluation
#         g, h = approx_tce(P, c, L, T)

#         # Entropic transition coupling improvement
#         P = entropic_tci2(h, P_old, Px, Py, xi, reg_num, sink_iter)

#     # In case of numerical instability, make non-negative and normalize.
#     P = np.maximum(P, 0)
#     row_sums = np.sum(P, axis=1, keepdims=True)
#     P = P / np.where(row_sums > 0, row_sums, 1)

#     if get_sd:
#         stat_dist, exp_cost = get_best_stat_dist(P,c)
#         stat_dist = np.reshape(stat_dist, (dx, dy))
#     else:
#         stat_dist = None
#         exp_cost = g[0].item()

#     return exp_cost, P, stat_dist
