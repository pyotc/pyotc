import numpy as np


def stochastic_block_model(sizes: tuple, probs: np.ndarray) -> np.ndarray:
    """Generate the adjacency for a stochastic block model SBM from a tuple (length n)
    of sizes an (nxn) matrix of probabilities.

    Args:
        sizes (tuple): tuple of node sizes with length of number of blocks
        probs (np.ndarray): nxn symmetric matrix

    Raises:
        ValueError: If probs is not a square numpy array
        ValueError: If probs is not symmetric
        ValueError: If sizes and probs dimensions do not match

    Returns:
        np.ndarray: adjancency matrix for SBM
    """
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
