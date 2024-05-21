import numpy as np
from numpy.linalg import pinv
from scipy.optimize import linprog
from scipy.linalg import fractional_matrix_power
import copy


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


def computeot_lp(C, r, c):
    nx = r.size
    ny = c.size
    Aeq = np.zeros((nx+ny, nx*ny))
    beq = np.concatenate((r.flatten(), c.flatten()))
    beq = beq.reshape(-1,1)

    # column sums correct
    for row in range(nx):
        for t in range(ny):
            Aeq[row, (row*ny)+t] = 1

    # row sums correct
    for row in range(nx, nx+ny):
        for t in range(nx):
            Aeq[row, t*ny+(row-nx)] = 1

    #lb = np.zeros(nx*ny)
    bound = [[0, None]] * (nx*ny)

    # solve OT LP using linprog
    cost = C.reshape(-1,1)
    res = linprog(cost, A_eq=Aeq, b_eq=beq, bounds=bound, method='highs-ipm')
    lp_sol = res.x
    lp_val = res.fun
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
                    sol, val = computeot_lp(g_mat, dist_x, dist_y)
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
                sol, val = computeot_lp(h_mat, dist_x, dist_y)
            idx = dy*(x_row)+y_row
            # print(x_row, y_row, P0[18, 1])
            Pz[idx, :] = np.reshape(sol, (-1, dx*dy))

    if np.max(np.abs(np.matmul(P0, h) - np.matmul(Pz, h))) <= 1e-4:
        Pz = copy.deepcopy(P0)
    return Pz


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

    # In case the solver fails due to numerical underflow, try with rescaling.
    # alpha = 1
    # while stat_dist.size == 0 and alpha >= 1e-10:
    #     alpha = alpha / 10
    #     res = linprog(c.flatten(), A_eq=alpha*Aeq, b_eq=alpha*beq.flatten(), bounds=(lb.flatten(), None), options=options)
    #     stat_dist = res.x.reshape((n, 1))
    #     exp_cost = res.fun
    # if stat_dist.size == 0:
    #     raise Exception('Failed to compute stationary distribution.')
    
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


def exact_otc1(Px, Py, c):
    dx = Px.shape[0]
    dy = Py.shape[0]

    P_old = np.ones((dx*dy, dx*dy))
    P = get_ind_tc(Px, Py)
    iter_ctr = 0
    while np.max(np.abs(P-P_old)) > 1e-10:
        #print(iter_ctr)
        iter_ctr += 1
        P_old = np.copy(P)

        # Transition coupling evaluation.
        g, h = exact_tce(P, c)

        # Transition coupling improvement.
        P = exact_tci(g, h, P_old, Px, Py)

        # Check for convergence.
        if np.all(P == P_old):
            stat_dist = get_stat_dist(P)
            stat_dist = np.reshape(stat_dist, (dx, dy))
            exp_cost = np.sum(stat_dist * c)
            return exp_cost, P, stat_dist

    return None, None, None
