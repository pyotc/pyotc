import numpy as np
from scipy.optimize import linprog


def get_best_stat_dist(P, c):
    """
    Given a transition matrix P and a cost vector c,
    this function computes the stationary distribution that minimizes the expected cost
    via linear programming.

    Args:
        P (np.ndarray): Transition matrix.
        c (np.ndarray): Cost vector.

    Returns:
        stat_dist (np.ndarray): Best stationary distribution.
        exp_cost (float): Corresponding expected cost.
    """

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


def get_stat_dist(P, method="best", c=None):
    """
    Computes the stationary distribution of a Markov chain given its transition matrix P.

    Supports multiple methods:
        - 'best': Solves a linear program that minimizes cost under stationarity constraints.
        - 'eigen': Solves for the stationary distribution using the eigenvalue method.
        - 'iterative': Uses power iteration for large or sparse matrices.

    Args:
        P (np.ndarray): Transition matrix of the Markov chain, shape (n, n).
        method (str): Method used to compute the stationary distribution.
                      One of 'eigen', 'iterative', or 'best'. Defaults to 'best'.
        c (np.ndarray, optional): Cost vector of shape (n,) used only when method='best'.

    Returns:
        pi (np.ndarray): Stationary distribution vector of shape (n,), summing to 1.

    Raises:
        ValueError: If method is 'best' but cost vector `c` is not provided,
                    or if an invalid method name is given.
    """
    if method == "best":
        # 'best' method minimizes expected cost under stationary constraints
        if c is None:
            raise ValueError("Cost function 'c' is required when method='best'.")

        n = P.shape[0]
        c = np.reshape(c, (n, -1))

        # Stationarity constraint: π^T P = π^T  ⇨  (P^T - I)^T π = 0
        # Add additional constraint: sum(π) = 1
        Aeq = np.concatenate((P.T - np.eye(n), np.ones((1, n))), axis=0)
        beq = np.concatenate((np.zeros((n, 1)), 1), axis=None)
        beq = beq.reshape(-1, 1)

        bound = [[0, None]] * n

        # Solve the linear program: minimize c^T π s.t. Aeq π = beq
        res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bound)
        pi = res.x
        return pi

    elif method == "eigen":
        # Computes the stationary distribution using eigenvalue decomposition
        # Calculate the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(P.T)

        # Identify the eigenvector associated with eigenvalue closest to 1 and normalize to obtain a valid distribution
        idx = np.argmin(np.abs(eigenvalues - 1))
        pi = np.real(eigenvectors[:, idx])
        pi /= np.sum(pi)
        return pi

    elif method == "iterative":
        # Computes the stationary distribution using power iteration
        max_iter = 10000
        tol = 1e-10
        n = P.shape[0]

        # Start from uniform distribution
        pi = np.ones(n) / n

        for _ in range(max_iter):
            pi_new = pi @ P
            if np.linalg.norm(pi_new - pi, ord=1) < tol:
                break
            pi = pi_new

        # Normalize the resulting distribution
        pi /= np.sum(pi)
        return pi

    else:
        raise ValueError(
            f"Invalid method '{method}'. Must be one of 'best', 'eigen', or 'iterative'."
        )
