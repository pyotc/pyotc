import numpy as np

def round_transpoly(X, r, c):
    A = X.copy()
    #A = copy.deepcopy(X)
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
    
    y = np.max(X, axis=axis, keepdims=True) #use 'keepdims' to make matrix operation X-y work
    s = y + np.log(np.sum(np.exp(X - y), axis=axis, keepdims=True))
    
    return np.squeeze(s, axis=axis)

def logsinkhorn(A, r, c, T):

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