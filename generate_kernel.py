# -*- coding: utf-8 -*-
# @Time    : 2021/12/16 9:14
# @Author  : Yizheng Wang
# @E-mail  : wyz020@126.com
# @File    : generate_kernel.py

import numpy as np

from hsic_kernel import hsic_kernel_m
from hsic_kernel import hsic_kernel_d

from kernel_normalized import kernel_normalized


def generate_kernel_m(cv):
    kernel_miRNA_gip = np.load('dataset/kernel/kernel_miRNA_gip,cv=' + str(cv) + '.npy')
    kernel_miRNA_seq = np.load('dataset/kernel/kernel_miRNA_seq.npy')
    kernel_miRNA_func = np.load('dataset/kernel/kernel_miRNA_func.npy')

    kernel_list = np.array([kernel_miRNA_gip,
                            kernel_miRNA_seq,
                            kernel_miRNA_func])

    return kernel_list


def generate_kernel_d(cv):
    kernel_disease_gip = np.load('dataset/kernel/kernel_disease_gip,cv=' + str(cv) + '.npy')
    kernel_disease_seq = np.load('dataset/kernel/kernel_disease_seq.npy')
    kernel_disease_func = np.load('dataset/kernel/kernel_disease_func.npy')
    kernel_list = np.array([kernel_disease_gip,
                            kernel_disease_seq,
                            kernel_disease_func])

    return kernel_list


def use_weight_combine_kernel_d(kernel_list_d, weight_d):
    weight_d = weight_d.reshape(weight_d.shape[0], 1, 1)
    kernel_d = kernel_list_d * weight_d
    kernel_d = np.sum(kernel_d, axis=0)

    return kernel_d


def use_weight_combine_kernel_s(kernel_list_s, weight_s):
    weight_s = weight_s.reshape(weight_s.shape[0], 1, 1)
    kernel_s = kernel_list_s * weight_s
    kernel_s = np.sum(kernel_s, axis=0)

    return kernel_s


if __name__ == "__main__":

    for i in range(5):
        nF = 'intermediate_result/F_asterisk,cv=' + str(i) + '.npy'
        nd = 'intermediate_result/kernel_m,cv=' + str(i) + '.npy'
        nt = 'intermediate_result/kernel_d,cv=' + str(i) + '.npy'
        F_asterisk = np.load(nF)

        kernel_list_d = generate_kernel_m(i)
        weight_d = hsic_kernel_m(kernel_list_d, F_asterisk, 0.01, 0.001)
        print(i, weight_d)
        d = use_weight_combine_kernel_d(kernel_list_d, weight_d)
        d = kernel_normalized(d)
        np.save(nd, d)

        kernel_list_t = generate_kernel_d(i)
        weight_t = hsic_kernel_d(kernel_list_t, F_asterisk, 0.01, 0.001)
        print(i, weight_t)
        t = use_weight_combine_kernel_s(kernel_list_t, weight_t)
        t = kernel_normalized(t)
        np.save(nt, t)


