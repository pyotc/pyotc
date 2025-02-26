import numpy as np
from numpy.linalg import pinv
from scipy.optimize import linprog
from scipy.linalg import fractional_matrix_power
import copy
import ot


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


def exact_tce(Pz, c):
    d = Pz.shape[0]
    c = np.reshape(c, (d, -1))
    A = np.block([[np.eye(d) - Pz, np.zeros((d, d)), np.zeros((d, d))],
                  [np.eye(d), np.eye(d) - Pz, np.zeros((d, d))],
                  [np.zeros((d, d)), np.eye(d), np.eye(d) - Pz]])
    b = np.concatenate([np.zeros((d, 1)), c, np.zeros((d, 1))])
    try:
        sol = np.linalg.solve(A, b)
    #except np.linalg.LinAlgError:
    except:
        sol = np.matmul(pinv(A), b)

    g = sol[0:d].flatten()
    h = sol[d:2*d].flatten()
    return g, h


def computeot_pot(C, r, c):
    # Ensure r and c are numpy arrays
    r = np.array(r).flatten()
    c = np.array(c).flatten()

    # Compute the optimal transport plan and the cost using the ot.emd function
    lp_sol = ot.emd(r, c, C)
    lp_val = np.sum(lp_sol * C)

    return lp_sol, lp_val


def exact_tci(g, h, P0, Px, Py):
    dx = Px.shape[0]
    dy = Py.shape[0]
    Pz = np.zeros((dx*dy, dx*dy))
    
    g_const = True
    for i in range(dx):
        for j in range(i+1, dx):
            if abs(g[i] - g[j]) > 1e-3:
                g_const = False
                break
        if not g_const:
            break
    # If g is not constant, improve transition coupling against g.
    if not g_const:
        g_mat = np.reshape(g, (dx, dy))
        for x_row in range(dx):
            for y_row in range(dy):
                dist_x = Px[x_row, :]
                dist_y = Py[y_row, :]
                # Check if either distribution is degenerate.
                if any(dist_x == 1) or any(dist_y == 1):
                    sol = np.outer(dist_x, dist_y)
                # If not degenerate, proceed with OT.
                else:
                    sol, val = computeot_pot(g_mat, dist_x, dist_y)
                idx = dy*(x_row)+y_row
                Pz[idx, :] = np.reshape(sol, (-1, dx*dy))
                #P[idx, :] = sol
        if np.max(np.abs(np.matmul(P0, g) - np.matmul(Pz, g))) <= 1e-7:
            Pz = copy.deepcopy(P0)
        else:
            return Pz
    ## Try to improve with respect to h.
    h_mat = np.reshape(h, (dx, dy))
    for x_row in range(dx):
        for y_row in range(dy):
            dist_x = Px[x_row, :]
            dist_y = Py[y_row, :]
            # Check if either distribution is degenerate.
            if any(dist_x == 1) or any(dist_y == 1):
                sol = np.outer(dist_x, dist_y)
            # If not degenerate, proceed with OT.
            else:
                sol, val = computeot_pot(h_mat, dist_x, dist_y)
            idx = dy*(x_row)+y_row
            # print(x_row, y_row, P0[18, 1])
            Pz[idx, :] = np.reshape(sol, (-1, dx*dy))

    if np.max(np.abs(np.matmul(P0, h) - np.matmul(Pz, h))) <= 1e-4:
        Pz = copy.deepcopy(P0)
    return Pz


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


def exact_otc(Px, Py, c):
    dx = Px.shape[0]
    dy = Py.shape[0]

    P_old = np.ones((dx*dy, dx*dy))
    P = get_ind_tc(Px, Py)
    iter_ctr = 0
    while np.max(np.abs(P-P_old)) > 1e-10:
        iter_ctr += 1
        P_old = np.copy(P)

        # Transition coupling evaluation.
        g, h = exact_tce(P, c)

        # Transition coupling improvement.
        P = exact_tci(g, h, P_old, Px, Py)

        # Check for convergence.
        if np.all(P == P_old):
            stat_dist, exp_cost = get_best_stat_dist(P,c)
            stat_dist = np.reshape(stat_dist, (dx, dy))
            return exp_cost, P, stat_dist

    return None, None, None