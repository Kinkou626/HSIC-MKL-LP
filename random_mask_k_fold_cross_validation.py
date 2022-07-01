# -*- coding: utf-8 -*-
# @Time    : 2021/12/31 8:06
# @Author  : Yizheng Wang
# @E-mail  : wyz020@126.com
# @File    : random_mask_k_fold_cross_validation.py


import numpy as np
import itertools
import random
import math


def random_mask_k_fold_cross_validation(matrix, k):
    random.seed(0)
    n = math.ceil((matrix.shape[0] * matrix.shape[1]) / k)
    random_list = []
    remain = list(itertools.product(range(matrix.shape[0]), range(matrix.shape[1])))
    for i in range(k-1):
        extracted = random.sample(remain, n)
        remain = list(set(remain).difference(set(extracted)))
        random_list.append(np.array(extracted))
    random_list.append(np.array(remain))

    mask_results = []
    mask_matrices = []
    for i in range(k):
        mask_result = np.zeros((random_list[i].shape[0], 3), dtype=int)
        mask_matrix = np.array(matrix)
        for j in range(random_list[i].shape[0]):
            random_row = random_list[i][j][0]
            random_column = random_list[i][j][1]
            mask_result[j, 0] = random_row
            mask_result[j, 1] = random_column
            mask_result[j, 2] = mask_matrix[random_row, random_column]
            mask_matrix[random_row, random_column] = 0
        mask_matrix = add_gaussian_noise(mask_matrix, 0.0000000005, 0.000000001)
        mask_results.append(mask_result)
        mask_matrices.append(mask_matrix)
    return mask_matrices, mask_results


def add_gaussian_noise(matrix, variance, mean):
    gaussian_noise = variance * np.random.randn(matrix.shape[0], matrix.shape[1]) + mean
    noise_matrix = matrix + gaussian_noise
    return noise_matrix


if __name__ == "__main__":
    adjacent_matrix = np.load('dataset/admat.npy')
    F_asterisk, mask = random_mask_k_fold_cross_validation(adjacent_matrix, 5)

    for i in range(5):
        n_F = 'intermediate_result/F_asterisk,cv=' + str(i)
        n_m = 'intermediate_result/mask,cv=' + str(i)
        np.save(n_F, F_asterisk[i])
        np.save(n_m, mask[i])

