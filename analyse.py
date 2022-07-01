# -*- coding: utf-8 -*-
# @Time    : 2021/12/27 7:57
# @Author  : Yizheng Wang
# @E-mail  : wyz020@126.com
# @File    : analyse.py

import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def calculate_AUPR(mask, F):
    y = mask[:, 2]
    y_p = np.ones(mask.shape[0])
    for j in range(y_p.shape[0]):
        y_p[j] = F[mask[j][0]][mask[j][1]]

    precision, recall, thresholds = precision_recall_curve(y, y_p)
    AUPR = auc(recall, precision)
    # print(len(precision), len(recall))
    return AUPR


def calculate_AUC(mask, F):
    y = mask[:, 2]
    y_p = np.ones(mask.shape[0])
    for j in range(y_p.shape[0]):
        y_p[j] = F[mask[j][0]][mask[j][1]]

    fpr, tpr, thresholds = roc_curve(y, y_p)
    AUC = auc(fpr, tpr)
    return AUC