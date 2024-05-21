import numpy as np
from .tce import exact_tce
from .tci import exact_tci
from scipy.optimize import linprog



def get_ind_tc(Px, Py):
    dx, dx_col = Px.shape
    dy, dy_col = Py.shape

    P_ind = np.zeros((dx*dy, dx_col*dy_col))
    for x_row in range(dx):
        for x_col in range(dx_col):
            for y_row in range(dy):
                for y_col in range(dy_col):
                    idx1 = dy*(x_row) + y_row
                    idx2 = dy*(x_col) + y_col
                    P_ind[idx1, idx2] = Px[x_row, x_col] * Py[y_row, y_col]
    return P_ind


def get_stat_dist(Pz):
    # Calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(Pz.T)

    # Find the index of the eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1))

    # Get the corresponding eigenvector
    stationary_dist = np.real(eigenvectors[:, idx])
    stationary_dist /= np.sum(stationary_dist)  # Normalize to make it a probability distribution

    return stationary_dist


def get_best_stat_dist(P, c):
    # Set up constraints.
    n = P.shape[0]
    c = np.reshape(c, (n, -1))
    Aeq = np.concatenate((P.T - np.eye(n), np.ones((1, n))), axis = 0)
    beq = np.concatenate((np.zeros((n, 1)), 1), axis = None)
    beq = beq.reshape(-1,1)
    bound = [[0, None]] * n
    
    # Solve linear program.
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bound)
    stat_dist = res.x
    exp_cost = res.fun
    return stat_dist, exp_cost

def exact_otc(Px, Py, c, tol=1e-10, ot_method="scipy"):
    dx = Px.shape[0]
    dy = Py.shape[0]

    P_old = np.ones((dx*dy, dx*dy))
    P = get_ind_tc(Px, Py)
    iter_ctr = 0
    while np.max(np.abs(P-P_old)) > tol:
        print(iter_ctr)
        iter_ctr += 1
        P_old = np.copy(P)

        # Transition coupling evaluation.
        g, h = exact_tce(P, c)

        # Transition coupling improvement.
        P = exact_tci(g, h, P_old, Px, Py, ot_method="scipy")

        # Check for convergence.
        if np.all(P == P_old):
            stat_dist = get_stat_dist(P)
            stat_dist = np.reshape(stat_dist, (dx, dy))
            exp_cost = np.sum(stat_dist * c)
            return iter_ctr, exp_cost, P, stat_dist

    return iter_ctr, P