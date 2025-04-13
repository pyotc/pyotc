import numpy as np
from scipy.optimize import linprog

# Check whether we can refactor the implementation using np.kron.
def get_ind_tc(Px, Py):
    """
    Computes the independent transition coupling of two transition matrices.

    Given two transition matrices Px and Py, this function returns the product
    transition matrix over the joint state space, assuming independence between
    the two chains.

    Args:
        Px (np.ndarray): Transition matrix of the first Markov chain of shape (dx, dx).
        Py (np.ndarray): Transition matrix of the second Markov chain of shape (dy, dy).

    Returns:
        np.ndarray: Independent transition coupling matrix of shape (dx * dy, dx * dy),
                    where entry (i, j) = Px[x_row, x_col] * Py[y_row, y_col].
    """
    dx, dx_col = Px.shape
    dy, dy_col = Py.shape

    P_ind = np.zeros((dx * dy, dx_col * dy_col))
    for x_row in range(dx):
        for x_col in range(dx_col):
            for y_row in range(dy):
                for y_col in range(dy_col):
                    idx1 = dy * (x_row) + y_row
                    idx2 = dy * (x_col) + y_col
                    P_ind[idx1, idx2] = Px[x_row, x_col] * Py[y_row, y_col]
    return P_ind
    

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
    Aeq = np.concatenate((P.T - np.eye(n), np.ones((1, n))), axis = 0)
    beq = np.concatenate((np.zeros((n, 1)), 1), axis = None)
    beq = beq.reshape(-1,1)
    bound = [[0, None]] * n
    
    # Solve linear program.
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bound)
    stat_dist = res.x
    exp_cost = res.fun
    
    return stat_dist, exp_cost


def get_stat_dist(P):
    """
    Computes the stationary distribution of a Markov chain given its transition matrix 
    using the eigenvalue method.

    Args:
        P (np.ndarray): Transition matrix of shape (n, n).

    Returns:
        stationary_dist (np.ndarray): Stationary distribution vector (shape: (n,)), normalized to sum to 1.
    """
    
    # Calculate the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(P.T)

    # Find the index of the eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1))

    # Get the corresponding eigenvector
    stationary_dist = np.real(eigenvectors[:, idx])
    stationary_dist /= np.sum(stationary_dist)  # Normalize to make it a probability distribution

    return stationary_dist

