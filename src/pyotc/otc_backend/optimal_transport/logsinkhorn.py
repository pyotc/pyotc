import numpy as np


def round_transpoly(X, r, c):
    A = X.copy()
    # A = copy.deepcopy(X)
    n1, n2 = A.shape

    r_A = np.sum(A, axis=1)
    for i in range(n1):
        scaling = min(1, r[i] / r_A[i])
        A[i, :] *= scaling

    c_A = np.sum(A, axis=0)
    for j in range(n2):
        scaling = min(1, c[j] / c_A[j])
        A[:, j] *= scaling

    r_A = np.sum(A, axis=1)
    c_A = np.sum(A, axis=0)
    err_r = r_A - r
    err_c = c_A - c

    if not np.all(err_r == 0) and not np.all(err_c == 0):
        A += np.outer(err_r, err_c) / np.sum(np.abs(err_r))

    return A


def logsumexp(X, axis=None):
    """
    Numerically stable log-sum-exp operation.

    Args:
        X (np.ndarray): Input array.
        axis (int or tuple of ints, optional): Axis or axes over which to operate.

    Returns:
        np.ndarray: The result of log(sum(exp(X))) along the specified axis.
    """

    y = np.max(
        X, axis=axis, keepdims=True
    )  # use 'keepdims' to make matrix operation X-y work
    s = y + np.log(np.sum(np.exp(X - y), axis=axis, keepdims=True))

    return np.squeeze(s, axis=axis)


def logsinkhorn(A, r, c, T):
    """
    Implementation of classical Sinkhorn algorithm for matrix scaling.
    Each iteration simply alternately updates (projects) all rows or
    all columns to have correct marginals.

    Args:
        A (np.ndarray): Negative scaled cost matrix of shape (dx, dy), e.g., -xi * cost.
        r (np.ndarray): desired row sums (marginals) (shape: dx,). Should sum to 1.
        c (np.ndarray): desired column sums (marginals) (shape: dy,). Should sum to 1.
        T (int): Number of full Sinkhorn iterations.

    Returns:
        np.ndarray: Final scaled matrix of shape (dx, dy).
    """

    dx, dy = A.shape
    f = np.zeros(dx)
    g = np.zeros(dy)

    for t in range(T):
        if t % 2 == 0:
            f = np.log(r) - logsumexp(A + g, axis=1)
        else:
            g = np.log(c) - logsumexp(A + f[:, np.newaxis], axis=0)

    P = round_transpoly(np.exp(f[:, np.newaxis] + A + g), r, c)

    return P
