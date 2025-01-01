import numpy as np


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


def get_degree_cost(D1, D2):
    d1 = D1.shape[0]
    d2 = D2.shape[0]
    degrees1 = np.sum(D1, axis=1)
    degrees2 = np.sum(D2, axis=1)
    cost_mat = np.zeros((d1, d2))
    for i in range(d1):
        for j in range(d2):
            cost_mat[i, j] = (degrees1[i] - degrees2[j]) ** 2
    return cost_mat


def get_01_cost(D1, D2):
    d1 = len(D1)
    d2 = len(D2)
    cost_mat = np.zeros((d1, d2))
    for i in range(d1):
        for j in range(d2):
            cost_mat[i, j] = D1[i] != D2[j]
    return cost_mat


def get_sq_cost(V1, V2):
    v1 = len(V1)
    v2 = len(V2)
    cost_mat = np.zeros((v1, v2))
    for i in range(v1):
        for j in range(v2):
            cost_mat[i, j] = (V1[i] - V2[j]) ** 2
    return cost_mat
