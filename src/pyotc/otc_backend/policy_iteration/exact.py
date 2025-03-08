import numpy as np
from .exact_tce import exact_tce
from .exact_tci import exact_tci
from .exact_tci_refactor import exact_tci as exact_tci_refactor
from .exact_tci_pot import exact_tci as exact_tci_pot
from .utils import get_ind_tc, get_stat_dist

def exact_otc1(Px, Py, c):
    dx = Px.shape[0]
    dy = Py.shape[0]

    P_old = np.ones((dx * dy, dx * dy))
    P = get_ind_tc(Px, Py)
    iter_ctr = 0
    while np.max(np.abs(P - P_old)) > 1e-10:
        # print(iter_ctr)
        iter_ctr += 1
        # P_old = P.copy()
        P_old = np.copy(P)
        # P_old = P

        # Transition coupling evaluation.
        g, h = exact_tce(P, c)

        # Transition coupling improvement.
        P = exact_tci(g, h, P_old, Px, Py)

        # Check for convergence.
        if np.all(P == P_old):
            stat_dist = get_stat_dist(P)
            # stat_dist = np.reshape(stat_dist, (dy, dx)).T
            stat_dist = np.reshape(stat_dist, (dx, dy))
            exp_cost = np.sum(stat_dist * c)
            return exp_cost, P, stat_dist

    return None, None, None


def exact_otc1_refactor(Px, Py, c):
    dx = Px.shape[0]
    dy = Py.shape[0]

    P_old = np.ones((dx * dy, dx * dy))
    P = get_ind_tc(Px, Py)
    iter_ctr = 0
    while np.max(np.abs(P - P_old)) > 1e-10:
        # print(iter_ctr)
        iter_ctr += 1
        # P_old = P.copy()
        P_old = np.copy(P)
        # P_old = P

        # Transition coupling evaluation.
        g, h = exact_tce(P, c)

        # Transition coupling improvement.
        P = exact_tci_refactor(g, h, P_old, Px, Py)

        # Check for convergence.
        if np.all(P == P_old):
            stat_dist = get_stat_dist(P)
            # stat_dist = np.reshape(stat_dist, (dy, dx)).T
            stat_dist = np.reshape(stat_dist, (dx, dy))
            exp_cost = np.sum(stat_dist * c)
            return exp_cost, P, stat_dist

    return None, None, None


def exact_otc1_pot(Px, Py, c):
    dx = Px.shape[0]
    dy = Py.shape[0]

    P_old = np.ones((dx * dy, dx * dy))
    P = get_ind_tc(Px, Py)
    iter_ctr = 0
    while np.max(np.abs(P - P_old)) > 1e-10:
        # print(iter_ctr)
        iter_ctr += 1
        # P_old = P.copy()
        P_old = np.copy(P)
        # P_old = P

        # Transition coupling evaluation.
        g, h = exact_tce(P, c)

        # Transition coupling improvement.
        P = exact_tci_pot(g, h, P_old, Px, Py)

        # Check for convergence.
        if np.all(P == P_old):
            stat_dist = get_stat_dist(P)
            # stat_dist = np.reshape(stat_dist, (dy, dx)).T
            stat_dist = np.reshape(stat_dist, (dx, dy))
            exp_cost = np.sum(stat_dist * c)
            return float(exp_cost), P, stat_dist

    return None, None, None
