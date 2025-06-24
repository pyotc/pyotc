import numpy as np


def weight(x):
    """
    Normalizes a vector into a probability distribution.

    Args:
        x (np.ndarray): Input vector.

    Returns:
        np.ndarray: Normalized vector such that the sum is 1.
    """
    return x / np.sum(x)


def adj_to_trans(A):
    """
    Converts an adjacency matrix into a row-stochastic transition matrix.

    Args:
        A (np.ndarray): Adjacency matrix of shape (n, n).

    Returns:
        np.ndarray: Transition matrix of shape (n, n), where each row sums to 1.
    """
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
    """
    Computes a cost matrix based on squared degree differences between nodes.

    Args:
        A1 (np.ndarray): First adjacency matrix of shape (n1, n1).
        A2 (np.ndarray): Second adjacency matrix of shape (n2, n2).

    Returns:
        cost_mat (np.ndarray): Cost matrix of shape (n1, n2) with squared degree differences.
    """
    n1 = A1.shape[0]
    n2 = A2.shape[0]
    degrees1 = np.sum(A1, axis=1)
    degrees2 = np.sum(A2, axis=1)
    cost_mat = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            cost_mat[i, j] = (degrees1[i] - degrees2[j]) ** 2
    return cost_mat


def get_01_cost(V1, V2):
    """
    Computes a binary cost matrix between node features of two graphs based on inequality.

    Given two vectors representing features of nodes from two graphs, this function
    returns a binary cost matrix where each entry is 1 if the corresponding features differ,
    and 0 otherwise.

    Args:
        V1 (np.ndarray): Feature vector for nodes in graph 1, of shape (n1,).
        V2 (np.ndarray): Feature vector for nodes in graph 2, of shape (n2,).

    Returns:
        np.ndarray: Binary cost matrix of shape (n1, n2), where entry (i, j) is 1
                    if V1[i] != V2[j], else 0.
    """

    n1 = len(V1)
    n2 = len(V2)
    cost_mat = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            cost_mat[i, j] = V1[i] != V2[j]
    return cost_mat


def get_sq_cost(V1, V2):
    """
    Computes a cost matrix based on squared differences between node features of two graphs.

    Given two vectors representing node features from two graphs, this function computes
    a cost matrix where each entry (i, j) is the squared difference between the i-th feature
    in graph 1 and the j-th feature in graph 2.

    Args:
        V1 (np.ndarray): Feature vector for nodes in graph 1, of shape (n1,).
        V2 (np.ndarray): Feature vector for nodes in graph 2, of shape (n2,).

    Returns:
        np.ndarray: Cost matrix of shape (n1, n2), where entry (i, j) = (V1[i] - V2[j]) ** 2.
    """

    n1 = len(V1)
    n2 = len(V2)
    cost_mat = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            cost_mat[i, j] = (V1[i] - V2[j]) ** 2
    return cost_mat
