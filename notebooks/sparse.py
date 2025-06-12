import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog 
import ot

import time


def weight(x):
    return x / np.sum(x)

def adj_to_trans(A):
    nrow = A.shape[0]
    T = np.copy(A).astype(float)
    for i in range(nrow):
        row = A[i, :]
        k = np.where(row != 0)[0]
        vals = weight(row[k])
        for idx in range(len(k)):
            T[i, k[idx]] = vals[idx]
    row_sums = T.sum(axis=1)
    return T / row_sums[:, np.newaxis]


def get_degree_cost(A1, A2):
    n1 = A1.shape[0]
    n2 = A2.shape[0]
    degrees1 = np.sum(A1, axis=1)
    degrees2 = np.sum(A2, axis=1)
    cost_mat = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            cost_mat[i, j] = (degrees1[i] - degrees2[j]) ** 2
    return cost_mat

def stochastic_block_model(sizes: tuple, probs: np.ndarray) -> np.ndarray:
    # Check input type
    if not isinstance(probs, np.ndarray) or probs.shape[0] != probs.shape[1]:
        raise ValueError("'probs' must be a square numpy array.")
    elif not np.allclose(probs, probs.T):
        raise ValueError("'probs' must be a symmetric matrix.")
    elif len(sizes) != probs.shape[0]:
        raise ValueError("'sizes' and 'probs' dimensions do not match.")

    n = sum(sizes)  # Total number of nodes
    n_b = len(sizes)  # Total number of blocks
    A = np.zeros((n, n))

    # Column index of each block's start
    cumsum = 0
    start = [0]
    for size in sizes:
        cumsum += size
        start.append(cumsum)

    # Generating Adjacency Matrix (upper)
    # Generate diagonal blocks
    for i in range(n_b):
        p = probs[i, i]
        for j in range(start[i], start[i + 1]):
            for k in range(j + 1, start[i + 1]):
                A[j, k] = np.random.choice([0, 1], p=[1 - p, p])

    # Generate Nondiagonal blocks
    for i in range(n_b - 1):
        for j in range(i + 1, n_b):
            A[start[i] : start[i + 1], start[j] : start[j + 1]] = np.random.choice(
                [0, 1], size=(sizes[i], sizes[j]), p=[1 - probs[i, j], probs[i, j]]
            )

    # Fill lower triangular matrix
    A = A + A.T

    return A

def computeot_pot(C, r, c):
    # Ensure r and c are numpy arrays
    r = np.array(r).flatten()
    c = np.array(c).flatten()

    # Compute the optimal transport plan and the cost using the ot.emd function
    lp_sol = ot.emd(r, c, C)
    lp_val = np.sum(lp_sol * C)

    return lp_sol, lp_val


def exact_tce_sparse(R_sparse, c):
    n = R_sparse.shape[0]
    c = np.reshape(c, (n, -1))
    I = sp.eye(n, format='csr')

    zero = sp.csr_matrix((n, n))
    A = sp.bmat([
        [I - R_sparse, zero, zero],
        [I, I - R_sparse, zero],
        [zero, I, I - R_sparse]
    ], format='csr')
    rhs = np.concatenate([np.zeros((n, 1)), c, np.zeros((n, 1))])

    # print("Solving sparse linear system in exact tce...")
    # permc_specs = ['COLAMD', 'MMD_ATA', 'MMD_AT_PLUS_A', 'NATURAL']
    # solution = None 
    # for spec in permc_specs:
    #     print(spec)
    #     try:
    #         current_solution = sp.linalg.spsolve(A, rhs, permc_spec=spec)
    #         print("spsolve successful with spec:", spec)
    #         if not np.any(np.abs(current_solution) > 1e15):
    #             solution = current_solution
    #             break 
    #         else:
    #             print(f"Solution with {spec} contains large values, trying next spec.")
    #     except ValueError as e:
    #         print(f"spsolve with {spec} encountered an error: trying next spec.")

    solution = sp.linalg.lsmr(A, rhs, atol=1e-10, btol=1e-10)[0]
    
    if solution is None:
        raise RuntimeError("Failed to find a stable solution with any of the provided permc_specs for sp.linalg.spsolve solver.")
    
    g = solution[:n]
    h = solution[n:2*n]
    return g, h

def setup_ot_sparse_fixed(f, Px, Py, Pz):
    dx = Px.shape[0]
    dy = Py.shape[0]
    f_mat = np.reshape(f, (dx, dy))

    for x_row in range(dx):
        for y_row in range(dy):
            dist_x = Px[x_row, :]
            dist_y = Py[y_row, :]
            # degenerate distribution check
            if np.any(dist_x == 1) or np.any(dist_y == 1):
                sol = np.outer(dist_x, dist_y)
            else:
                sol, _ = computeot_pot(f_mat, dist_x, dist_y) 
            idx = dy * x_row + y_row
            sol_flat = sol.flatten()
            for j in np.nonzero(sol_flat)[0]:
                Pz[idx, j] = sol_flat[j]
    return Pz

def exact_tci_sparse(g, h, P0, Px, Py): 
    dx, dy = Px.shape[0], Py.shape[0]
    Pz = sp.lil_matrix((dx * dy, dx * dy))
    g_const = np.max(g) - np.min(g) <= 1e-3

    if not g_const:
        Pz = setup_ot_sparse_fixed(g, Px, Py, Pz) 
        if np.max(np.abs(P0.dot(g) - Pz.dot(g))) <= 1e-7:
            Pz = P0.copy()
        else:
            return Pz

    Pz = setup_ot_sparse_fixed(h, Px, Py, Pz) 
    if np.max(np.abs(P0.dot(h) - Pz.dot(h))) <= 1e-4:
        Pz = P0.copy()

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


def get_best_stat_dist_sparse(P_sparse, c):
    n = P_sparse.shape[0]
    c = np.reshape(c, (n, -1))

    # Construct Aeq in sparse format
    eye_n = sp.eye(n, format='csr')
    row_sum = sp.csr_matrix(np.ones((1, n)))  # sum(x) = 1 constraint
    Aeq_sparse = sp.vstack([P_sparse.transpose() - eye_n, row_sum], format='csr')
    beq = np.concatenate((np.zeros((n, 1)), [[1]]), axis=0)
    bounds = [(0, None)] * n

    # Solve linear program (method='highs' supports sparse)
    res = linprog(c,
                A_eq=Aeq_sparse,
                b_eq=beq,
                bounds=bounds,
                method='highs')
    if not res.success:
        raise RuntimeError("Linear program failed: " + res.message)

    return res.x, res.fun

def get_stat_dist_sparse(P_sparse, max_iter=10000, tol=1e-10):
    """
    Computes the stationary distribution of a sparse transition matrix using power iteration.

    Args:
        P_sparse (sp.spmatrix): (n x n) row-stochastic transition matrix
        max_iter (int): max number of iterations
        tol (float): convergence tolerance

    Returns:
        pi (np.ndarray): stationary distribution of shape (n,)
    """
    n = P_sparse.shape[0]
    pi = np.ones(n) / n  # initial uniform distribution

    for _ in range(max_iter):
        pi_new = pi @ P_sparse
        if np.linalg.norm(pi_new - pi, ord=1) < tol:
            break
        pi = pi_new

    pi /= np.sum(pi)  # ensure normalization
    return pi



def exact_otc_pot_sparse(Px, Py, c, get_best_sd=True, max_iter=100):
    start = time.time()
    
    print("Starting exact_otc_pot_sparse...")
    dx, dy = Px.shape[0], Py.shape[0]
    P = sp.kron(sp.csr_matrix(Px), sp.csr_matrix(Py), format='csr')
    
    for iter in range(max_iter):
        print("Iteration:", iter)

        P_old = P.copy()
        
        print("Computing exact TCE...")
        g, h = exact_tce_sparse(P, c)
        
        print("Computing exact TCI...")
        P = exact_tci_sparse(g, h, P_old, Px, Py) #, forbidden_set)

        #print(np.max(np.abs(P.toarray()-P_old.toarray())))
        #if np.max(np.abs(P.toarray()-P_old.toarray())) <= 1e-10:
        if (P != P_old).nnz == 0:
            print("Convergence reached. Computing stationary distribution...")
            if get_best_sd:
                #stat_dist, exp_cost = get_best_stat_dist(P,c)
                stat_dist, exp_cost = get_best_stat_dist_sparse(P, c)
                stat_dist = np.reshape(stat_dist, (dx, dy))
            else:
                stat_dist = get_stat_dist_sparse(P)
                stat_dist = np.reshape(stat_dist, (dx, dy))
                exp_cost = g[0].item()
            end = time.time()
            print(f"Convergence reached in {iter} iterations, took {end - start:.4f} seconds")
            return float(exp_cost), P, stat_dist

    return None, None, None



if __name__ == "__main__":
    # Seed number
    np.random.seed(1004)
    
    m = 40
    A1 = stochastic_block_model(
        (m, m, m, m),
        np.array(
            [
                [0.9, 0.1, 0.1, 0.1],
                [0.1, 0.9, 0.1, 0.1],
                [0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.9],
            ]
        ),
    )
    A2 = stochastic_block_model(
        (m, m, m, m),
        np.array(
            [
                [0.9, 0.1, 0.1, 0.1],
                [0.1, 0.9, 0.1, 0.1],
                [0.1, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.9],
            ]
        ),
    )
    P1 = adj_to_trans(A1)
    P2 = adj_to_trans(A2)
    c = get_degree_cost(A1, A2)

    start = time.time()
    exp_cost, otc, stat_dist = exact_otc_pot_sparse(P1, P2, c)
    end = time.time()
    print("Cost:", exp_cost)
    print("Time:", end - start)

