import numpy as np
from .utils import get_ind_tc, get_best_stat_dist
from .approx_tce import approx_tce
from .entropic_tci import entropic_tci, entropic_tci1


def entropic_otc(Px, Py, c, L = 100, T = 100, xi = 0.1, sink_iter = 10, get_sd = False):

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
        stat_dist, exp_cost = get_best_stat_dist(P,c)
        stat_dist = np.reshape(stat_dist, (dx, dy))
    else:
        stat_dist = None
        exp_cost = g[0].item()

    return exp_cost, P, stat_dist


def entropic_otc1(Px, Py, c, L = 100, T = 100, xi = 0.1, reg_num = 0.1, sink_iter = 10, get_sd = False):

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
        stat_dist, exp_cost = get_best_stat_dist(P,c)
        stat_dist = np.reshape(stat_dist, (dx, dy))
    else:
        stat_dist = None
        exp_cost = g[0].item()

    return exp_cost, P, stat_dist