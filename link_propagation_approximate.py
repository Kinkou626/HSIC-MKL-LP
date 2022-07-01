# -*- coding: utf-8 -*-
# @Time    : 2021/12/5 14:09
# @Author  : Yizheng Wang
# @E-mail  : wyz020@126.com
# @File    : linkPropagationApproximate.py
import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import fractional_matrix_power
from analyse import calculate_AUC
from analyse import calculate_AUPR


# def cholesky_decomposition(A):
#     G = np.zeros((A.shape[0], A.shape[1]), dtype=float)
#     for j in range(1, A.shape[0] + 1):
#         for i in range(1, j):
#             sum_i = 0
#             for k in range(1, i):
#                 sum_i = sum_i + G[k - 1][i - 1] * G[k - 1][j - 1]
#             G[i - 1][j - 1] = (A[i - 1][j - 1] - sum_i) / G[i - 1][i - 1]
#         sum_k = 0
#         for k in range(1, j):
#             sum_k = sum_k + (G[k - 1][j - 1] ** 2)
#         G[j - 1][j - 1] = (A[j - 1][j - 1] - sum_k) ** 0.5
#     return np.transpose(G)
#
#
# def imaginary_matrix_to_real_matrix(im):
#     """
#     Transform an imaginary matrix into a real matrix (supports 1 and 2 dimensional arrays)
#     :param im: imaginary matrix of input
#     :return: real matrix
#     """
#     if im.ndim == 2:
#         rm = np.zeros((im.shape[0], im.shape[1]), dtype='float64')
#         for i in range(im.shape[0]):
#             for j in range(im.shape[1]):
#                 rm[i][j] = im[i][j].real
#         return rm
#     else:
#         rm = np.zeros((im.shape[0]), dtype='float64')
#         for i in range(im.shape[0]):
#             rm[i] = im[i].real
#         return rm


def link_propagation_approximate(W_X, W_Y, F_asterisk, kronecker_flag, sigma, kr1, kr2):
    # 1.Compute the low-rank approximation matrices G_X and G_Y of W_X and W_Y
    u, s, v = svds(W_X, kr1)
    G_X = np.matmul(u, np.diag(s ** 0.5))

    u, s, v = svds(W_Y, kr2)
    G_Y = np.matmul(u, np.diag(s ** 0.5))

    # 2.Compute the normalized matrices G_X_tilde and G_Y_tilde
    W_X_approximate = np.matmul(G_X, np.transpose(G_X))
    W_X_rows_sum = np.sum(W_X_approximate, axis=0)
    D_X = np.diag(W_X_rows_sum)
    G_X_tilde = fractional_matrix_power(D_X, -0.5)  # D_X to the minus 1/2
    G_X_tilde = np.matmul(G_X_tilde, G_X)

    W_Y_approximate = np.matmul(G_Y, np.transpose(G_Y))
    W_Y_rows_sum = np.sum(W_Y_approximate, 0)
    D_Y = np.diag(W_Y_rows_sum)
    G_Y_tilde = fractional_matrix_power(D_Y, -0.5)  # D_Y to the minus 1/2
    G_Y_tilde = np.matmul(G_Y_tilde, G_Y)

    # 3.Compute the eigendecomposition
    lambda_X_bar, U_X_bar = np.linalg.eig(np.matmul(np.transpose(G_X_tilde), G_X_tilde))

    lambda_Y_bar, U_Y_bar = np.linalg.eig(np.matmul(np.transpose(G_Y_tilde), G_Y_tilde))

    # 4.Compute the eigenvectors
    V_X_bar = np.matmul(G_X_tilde, U_X_bar)
    V_X_bar = np.matmul(V_X_bar, np.diag(lambda_X_bar ** -0.5))

    V_Y_bar = np.matmul(G_Y_tilde, U_Y_bar)
    V_Y_bar = np.matmul(V_Y_bar, np.diag(lambda_Y_bar ** -0.5))

    # 5.Compute the elements of the matrix D_bar
    V_inverted_bar = np.ones((kr1, kr2))
    if kronecker_flag == 0:
        for i in range(V_inverted_bar.shape[0]):
            for j in range(V_inverted_bar.shape[1]):
                V_inverted_bar[i][j] = lambda_X_bar[i] * lambda_Y_bar[j]
    else:
        for i in range(V_inverted_bar.shape[0]):
            for j in range(V_inverted_bar.shape[1]):
                V_inverted_bar[i][j] = lambda_X_bar[i] + lambda_Y_bar[j]

    if kronecker_flag == 0:
        c = 1
    else:
        c = 3

    D_bar = np.ones((kr1, kr2))
    for i in range(D_bar.shape[0]):
        for j in range(D_bar.shape[1]):
            D_bar[i][j] = sigma * (1 + c * sigma) * V_inverted_bar[i][j] / (
                    1 + c * sigma - sigma * V_inverted_bar[i][j])

    # 6.Compute the elements of the matrix F
    F = np.matmul(np.transpose(V_X_bar), F_asterisk)
    F = np.matmul(F, V_Y_bar)
    F = D_bar * F
    F = np.matmul(V_X_bar, F)
    F = np.matmul(F, np.transpose(V_Y_bar))
    F = (1 / (1 + c * sigma) ** 2) * F
    F = (1 / (1 + c * sigma)) * F_asterisk + F

    return F


if __name__ == "__main__":

    sigma = 0.2
    kr1 = 175
    kr2 = 325

    AUPR_sum = 0
    AUC_sum = 0
    for i in range(5):
        n_kernel_m = 'intermediate_result/kernel_m,cv=' + str(i) + '.npy'
        n_kernel_d = 'intermediate_result/kernel_d,cv=' + str(i) + '.npy'
        n_F_asterisk = 'intermediate_result/F_asterisk,cv=' + str(i) + '.npy'
        n_mask = 'intermediate_result/mask,cv=' + str(i) + '.npy'
        r = link_propagation_approximate(W_X=np.load(n_kernel_m),
                                         W_Y=np.load(n_kernel_d),
                                         F_asterisk=np.load(n_F_asterisk),
                                         kronecker_flag=1,  # 0 is Kronecker product and 1 is for Kronecker sum
                                         sigma=sigma,
                                         kr1=kr1,
                                         kr2=kr2)
        AUC = calculate_AUC(np.load(n_mask), r)
        AUC_sum = AUC_sum + AUC
        AUPR = calculate_AUPR(np.load(n_mask), r)
        AUPR_sum = AUPR_sum + AUPR
        n_r = 'intermediate_result/r,cv=' + str(i) + '.npy'
        np.save(n_r, r)
        print(i, AUPR, AUC)
    print(sigma, kr1, kr2, AUPR_sum / 5, AUC_sum/5)
