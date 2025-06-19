import numpy as np
import scipy.sparse as sp

def exact_tce(R_sparse, c):
    '''
    Computes the exact TCE (Transport Cost Estimation) for a sparse transition matrix R_sparse and cost vector c.

    Solving Ax = b using a direct solver (sp.linalg.spsolve) on large networks resulted in:
    "Not enough memory to perform factorization."
    This is likely due to excessive fill-in during LU factorization of the large sparse matrix.

    To address this, we switch to an iterative solver (scipy.sparse.linalg.lsmr),
    which is more memory-efficient and better suited for large-scale sparse systems.
    '''
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
    #     try:
    #         current_solution = sp.linalg.spsolve(A, rhs, permc_spec=spec)
    #         if not np.any(np.abs(current_solution) > 1e15):
    #         print("spsolve successful with spec:", spec)
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
