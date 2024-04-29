import networkx as nx
import numpy as np

def stochastic_block_model(sizes, probs):
    # Check input type
    if not isinstance(sizes, np.ndarray) or len(sizes.shape) != 1:
        raise ValueError("'sizes' must be a 1D numpy array.")
    elif not isinstance(probs, np.ndarray) or probs.shape[0] != probs.shape[1]:
        raise ValueError("'probs' must be a square numpy array.")
    elif not np.allclose(probs, probs.T):
        raise ValueError("'probs' must be a symmetric matrix.")
    elif len(sizes) != probs.shape[0]:
        raise ValueError("'sizes' and 'probs' dimensions do not match.")

    n = np.sum(sizes)  # Total number of nodes
    n_b = len(sizes)  # Total number of blocks
    A = np.zeros((n, n))

    # Column index of each block's start
    start = [0] + list(np.cumsum(sizes))

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
            A[start[i]:start[i + 1], start[j]:start[j + 1]] = np.random.choice([0, 1], size=(sizes[i], sizes[j]),
                                                                               p=[1 - probs[i, j], probs[i, j]])

    # Fill lower triangular matrix
    A = A + A.T

    return A


# Seed number
np.random.seed(10)

m = 10
A1 = stochastic_block_model(np.array([m,m,m,m]), np.array([[0.9,0.1,0.1,0.1],[0.1,0.9,0.1,0.1],[0.1,0.1,0.9,0.1],[0.1,0.1,0.1,0.9]]))

# Adjacency matrix
A2 = A1.copy()
A2[0, 23] = 0
A2[23, 0] = 0
A2[3, 24] = 0
A2[24, 3] = 0

A3 = A1.copy()
A3[0, 1] = 0
A3[1, 0] = 0
A3[3, 4] = 0
A3[4, 3] = 0

sbm_1 = nx.from_numpy_array(A1)
sbm_2 = nx.from_numpy_array(A2)
sbm_3 = nx.from_numpy_array(A3)