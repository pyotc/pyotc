import collections
import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))


def weight(x):
    return x / np.sum(x)


def adj_to_trans1(A):
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


def adj_to_trans2(A):
    nrow = A.shape[0]
    T = np.copy(A).astype(float)
    for i in range(nrow):
        row = A[i, :]
        k = np.where(row != 0)[0]
        vals = softmax(row[k])
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


def strength_cost(T1, T2):
    d1 = T1.shape[0]
    d2 = T2.shape[0]
    cost_mat = np.zeros((d1, d2))
    for i in range(d1):
        for j in range(d2):
            cost_mat[i, j] = (T2[j, j] - T1[i, i]) ** 2 + (
                (sum(T1[:, i]) - T1[i, i]) - (sum(T2[:, j]) - T2[j, j])
            ) ** 2
    return cost_mat


def extend_discrete_P(P, feature, alpha=0.5):
    count = collections.Counter(feature)
    P_new = np.zeros((len(feature), len(feature)))
    for idx1 in range(len(feature)):
        for idx2 in range(len(feature)):
            if feature[idx1] == feature[idx2]:
                P_new[idx1, idx2] = 1 / count[feature[idx1]]
    P_ext = (1 - alpha) * P + alpha * P_new
    row_sums = P_ext.sum(axis=1)
    return P_ext / row_sums[:, np.newaxis]


def extend_continuous_P(P, feature, alpha=0.5):
    D = np.zeros((len(feature), len(feature)))
    for i in range(0, len(feature)):
        for j in range(i + 1, len(feature)):
            # D[i,j] = np.exp(-(feature[i]-feature[j])**2/2)
            D[i, j] = 1 / (np.abs(feature[i] - feature[j]) + 1)
    D_new = D + D.T
    row_sums1 = D_new.sum(axis=1)
    P_new = D_new / row_sums1[:, np.newaxis]
    P_ext = (1 - alpha) * P + alpha * P_new
    row_sums2 = P_ext.sum(axis=1)
    return P_ext / row_sums2[:, np.newaxis]


def round_transition_matrix(matrix, n=5):
    rounded_matrix = np.round(matrix, n)
    # Adjust each row to ensure row sums equal to 1
    for i in range(rounded_matrix.shape[0]):
        row_sum = np.sum(rounded_matrix[i])
        if row_sum != 1:
            difference = 1 - row_sum
            # Find the index of the largest element in the row
            max_index = np.argmax(rounded_matrix[i])
            # Adjust the largest element by the difference
            rounded_matrix[i, max_index] += difference
    return rounded_matrix
