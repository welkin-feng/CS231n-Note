#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   CS231n-Note 

File Name:  Loss.py

"""
import numpy as np

__author__ = 'Welkin'
__date__ = '2017/8/10 15:15'

def L_i(x, y, W):
    """
    对于给定的一个样本数据(x, y)，算出 W 针对这个数据 x 在其正确分类 y 上的 SVM 损失
    unvectorized version. Compute the multiclass svm loss for a single example (x,y)
    - x is a column vector representing an image (e.g. 3073 x 1 in CIFAR-10)
      with an appended bias dimension in the 3073-rd position (i.e. bias trick)
    - y is an integer giving index of correct class (e.g. between 0 and 9 in CIFAR-10)
    - W is the weight matrix (e.g. 10 x 3073 in CIFAR-10)

    Li = ∑(j ≠ y_i) max(0, s_i - s_y_i + Δ)
    """
    delta = 1.0  # see notes about delta later in this section
    scores = W.dot(x)  # scores becomes of size 10 x 1, the scores for each class
    correct_class_score = scores[y]
    D = W.shape[0]  # number of classes, e.g. 10
    loss_i = 0.0
    for j in range(D):  # iterate over all wrong classes
        if j == y:
            # skip for the true class to only loop over incorrect classes
            continue
        # accumulate loss for the i-th example
        loss_i += max(0, scores[j] - correct_class_score + delta)
    return loss_i

def L_i_vectorized(x, y, W):
    """
    对于单个样本数据(x, y)，使用矩阵运算代替循环，加快运算速度，但是在这个函数外依旧需要一个循环
    A faster half-vectorized implementation. half-vectorized
    refers to the fact that for a single example the implementation contains
    no for loops, but there is still one loop over the examples (outside this function)
    """
    delta = 1.0
    scores = W.dot(x)
    # compute the margins for all classes in one vector operation
    margins = np.maximum(0, scores - scores[y] + delta)
    # on y-th position scores[y] - scores[y] canceled and gave delta. We want
    # to ignore the y-th position and only consider margin on max wrong class
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i

def Regular(W, lam, L = 'L2'):
    if L == 'L2':
        return lam * np.sum(np.square(W))
    elif L == 'L1':
        return lam * np.sum(np.abs(W))
    else:
        raise ValueError("The 'L' must be either 'L1' or 'L2")

def L_SVM(X, y, W, delta = 1.0, lam = 0.1):
    """
    fully-vectorized implementation :
    - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
    - y is array of integers specifying correct class (e.g. 50,000-D array)
    - W are weights (e.g. 10 x 3073)

    L = 1/N·∑(i=1, N)∑(j≠y_i)[max(0, f(x_i; W)_j - f(x_i; W)_y_i + Δ) + λ·R(W)
    """
    # evaluate loss over all examples in X without using any for loops
    # left as exercise to reader in the assignment
    N = y.shape[0]
    if X.shape[0] == N:
        X = X.T
    elif len(X.shape) < 1:
        X = X.reshape((N, -1)).T
    elif X.shape[1] == N:
        pass
    else:
        raise ValueError("The size of X is not matching the size of y")

    if W.shape[1] != X.shape[0]:
        raise ValueError("The sizes of W and X do not match each other")

    scores = W.dot(X)  # e.g. 10 x 50000
    margins = np.maximum(0, scores - scores[y, range(N)] + delta)
    margins[y, range(N)] = 0  # e.g. 10 x 50000
    loss = np.sum(margins) / N + Regular(W, lam)
    return loss

def L_softmax(X, y, W, lam = 1.0):
    N = y.shape[0]
    if X.shape[0] == N:
        X = X.T
    elif len(X.shape) < 1:
        X = X.reshape((N, -1)).T
    elif X.shape[1] == N:
        pass
    else:
        raise ValueError("The size of X is not matching the size of y")

    if W.shape[1] != X.shape[0]:
        raise ValueError("The sizes of W and X do not match each other")

    scores = W.dot(X)  # e.g. 10 x 50000
    scores = scores - np.max(scores, axis = 0)
    softmax = np.exp(scores) / np.sum(np.exp(scores), axis = 0)
    margins = softmax[y, range(N)]
    loss = np.sum(- np.log(margins)) / N + Regular(W, lam)
    return loss


if __name__ == '__main__':
    pass
