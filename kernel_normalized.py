# -*- coding: utf-8 -*-
# @Time    : 2022/1/8 10:01
# @Author  : Yizheng Wang
# @E-mail  : wyz020@126.com
# @File    : kernel_normalized.py

import numpy as np


def kernel_normalized(kernel):
    r = np.ones((kernel.shape[0], kernel.shape[1]))
    rows_sum = np.sum(kernel, axis=1)
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            r[i][j] = kernel[i][j] / rows_sum[i]
    for k in range(kernel.shape[0]):
        r[k][k] = 1
    return r
