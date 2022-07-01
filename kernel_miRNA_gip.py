# -*- coding: utf-8 -*-
# @Time    : 2021/11/13 21:15
# @Author  : Yizheng Wang
# @Email   : wyz020@126.com
# @File    : kernel_GIP_link_d_s.py

import numpy as np
import math
import scipy.io


def kernel_GIP_link_d(F_asterisk):
    F_asterisk = np.array(F_asterisk, dtype=float)
    kernel_matrix = np.ones((F_asterisk.shape[0], F_asterisk.shape[0]))
    for i in range(kernel_matrix.shape[0]):
        for j in range(kernel_matrix.shape[1]):
            if j < i:
                kernel_matrix[i][j] = kernel_matrix[j][i]
            else:
                r = F_asterisk[i] - F_asterisk[j]
                kernel_matrix[i][j] = math.exp(-1 * (np.linalg.norm(r, 2) ** 2))

    return kernel_matrix


if __name__ == "__main__":
    for i in range(5):
        nF = 'intermediate_result/F_asterisk,cv=' + str(i)
        nk = 'dataset/kernel/kernel_miRNA_gip,cv=' + str(i)
        F_asterisk = np.load(nF + '.npy')
        kernel = kernel_GIP_link_d(F_asterisk)
        np.save(nk, kernel)

