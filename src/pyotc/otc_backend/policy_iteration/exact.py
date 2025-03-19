import numpy as np
from .exact_tce import exact_tce
from .exact_tci_lp import exact_tci as exact_tci_lp
from .exact_tci_pot import exact_tci as exact_tci_pot
from .utils import get_ind_tc, get_stat_dist, get_best_stat_dist


def exact_otc_lp(Px, Py, c, get_best_sd=True):
    dx = Px.shape[0]
    dy = Py.shape[0]

    P_old = np.ones((dx * dy, dx * dy))
    P = get_ind_tc(Px, Py)
    while np.max(np.abs(P - P_old)) > 1e-10:
        P_old = np.copy(P)

        # Transition coupling evaluation.
        g, h = exact_tce(P, c)

        # Transition coupling improvement.
        P = exact_tci_lp(g, h, P_old, Px, Py)

        # Check for convergence.
        if np.all(P == P_old):
            if get_best_sd:
                stat_dist, exp_cost = get_best_stat_dist(P, c)
                stat_dist = np.reshape(stat_dist, (dx, dy))                
            else:         
                stat_dist = get_stat_dist(P)
                stat_dist = np.reshape(stat_dist, (dx, dy))
                exp_cost = np.sum(stat_dist * c)
            return float(exp_cost), P, stat_dist

    return None, None, None


def exact_otc_pot(Px, Py, c, get_best_sd=True):
    dx = Px.shape[0]
    dy = Py.shape[0]

    P_old = np.ones((dx * dy, dx * dy))
    P = get_ind_tc(Px, Py)
    while np.max(np.abs(P - P_old)) > 1e-10:
        P_old = np.copy(P)

        # Transition coupling evaluation.
        g, h = exact_tce(P, c)

        # Transition coupling improvement.
        P = exact_tci_pot(g, h, P_old, Px, Py)

        # Check for convergence.
        if np.all(P == P_old):
            if get_best_sd:
                stat_dist, exp_cost = get_best_stat_dist(P, c)
                stat_dist = np.reshape(stat_dist, (dx, dy))                
            else:         
                stat_dist = get_stat_dist(P)
                stat_dist = np.reshape(stat_dist, (dx, dy))
                exp_cost = np.sum(stat_dist * c)
            return float(exp_cost), P, stat_dist

    return None, None, None
