"""
Original Transition Coupling Evaluation (TCE) method from:
https://www.jmlr.org/papers/volume23/21-0519/21-0519.pdf
"""

import numpy as np
import scipy.sparse as sp


def exact_tce(R_sparse, c):
    """
    Computes the exact Transition Coupling Evaluation (TCE) vectors g and h for a given sparse transition matrix R_sparse and cost vector c.

    Specifically, solves the block linear system outlined in Algorithm 1a of the paper:
    "Optimal Transport for Stationary Markov Chains via Policy Iteration"
    (https://www.jmlr.org/papers/volume23/21-0519/21-0519.pdf).

    Due to memory constraints associated with direct solvers (e.g., sp.linalg.spsolve),
    an iterative solver (scipy.sparse.linalg.lsmr) is employed to efficiently handle large-scale sparse systems.

    Notes:
        1. When A in Ax = b is close to singular, we have observed few cases that both SciPy functions (scipy.sparse.linalg.spsolve, scipy.sparse.linalg.lsmr)
        can produce results that differ from NumPy's solver, leading to different results with dense implementation and non-convergence.
        This is an issue with SciPy solvers and remains an unresolved issue. The best approach in such cases is to fall back to the dense implementation.

        2. Solving Ax = b using a direct solver (scipy.sparse.linalg.spsolve) on large networks resulted in:
        "Not enough memory to perform factorization."
        This is likely due to excessive fill-in during LU factorization of the large sparse matrix.
        To address this, we switch to an iterative solver (scipy.sparse.linalg.lsmr),
        which is more memory-efficient and better suited for large-scale sparse systems.

        3. To leave open the possibility of switching from 'lsmr' to 'spsolve', the corresponding 'spsolve' code has been retained as a commented-out block.

    Args:
        R_sparse (scipy.sparse.csr_matrix): Sparse transition matrix of shape (dx*dy, dx*dy).
        c (np.ndarray): Cost vector of shape (dx, dy).

    Returns:
        g (np.ndarray): Average cost (gain) vector of shape (dx*dy,).
        h (np.ndarray): Total extra cost (bias) vector of shape (dx*dy,).
    """
    n = R_sparse.shape[0]
    c = c.reshape(n)

    # Construct the block matrix A and right-hand side vector b
    I = sp.eye(n, format="csr")
    zero = sp.csr_matrix((n, n))
    A = sp.bmat(
        [[I - R_sparse, zero, zero], [I, I - R_sparse, zero], [zero, I, I - R_sparse]],
        format="csr",
    )
    b = np.concatenate([np.zeros(n), c, np.zeros(n)])

    # print("Solving sparse linear system in exact tce...")
    # permc_specs = ['COLAMD', 'MMD_ATA', 'MMD_AT_PLUS_A', 'NATURAL']
    # solution = None
    # for spec in permc_specs:
    #     try:
    #         current_solution = sp.linalg.spsolve(A, rhs, permc_spec=spec)
    #         if not np.any(np.abs(current_solution) > 1e15):
    #             print("spsolve successful with spec:", spec)
    #             solution = current_solution
    #             break
    #         else:
    #             print(f"Solution with {spec} contains large values, trying next spec.")
    #     except ValueError as e:
    #         print(f"spsolve with {spec} encountered an error: trying next spec.")
    # if solution is None:
    #     raise RuntimeError("Failed to find a stable solution with any of the provided permc_specs for sp.linalg.spsolve solver.")

    # Solve the linear system using an iterative solver (lsmr)
    solution = sp.linalg.lsmr(A, b, atol=1e-10, btol=1e-10)[0]

    # Extract vectors g and h from the solution
    g = solution[:n]
    h = solution[n : 2 * n]

    return g, h
