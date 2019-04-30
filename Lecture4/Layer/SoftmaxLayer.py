#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   CS231n-Note 

File Name:  SoftmaxLayer.py

"""
import numpy as np

__author__ = 'Welkin'
__date__ = '2017/8/13 20:33'


class SoftmaxLayer(object):
    def __init__(self, W, X, y, lam = 0.1):
        self.W = np.copy(W)
        self.X = np.copy(X)
        self.y = np.copy(y)
        self.lam = lam


    def forward(self, data = None, weights = None, lam = None):
        if weights is not None:
            self.W = np.copy(weights)
        if data is not None:
            self.X, self.y = data
        if lam is not None:
            self.lam = lam

        N = self.y.shape[0]
        if self.X.shape[0] == N:
            X = self.X.T
        elif len(self.X.shape) < 1:
            X = self.X.reshape((N, -1)).T
        elif self.X.shape[1] == N:
            X = self.X
        else:
            raise ValueError("The size of X is not matching the size of y")

        if self.W.shape[1] != X.shape[0]:
            raise ValueError("The sizes of W and X do not match each other")

        W = self.W

        scores = W.dot(X)  # e.g. 10 x 50000
        scores = scores - np.max(scores, axis = 0)

        # 对应所有类别的概率
        self.softmax = np.exp(scores) / np.sum(np.exp(scores), axis = 0)
        # 对应正确类别的概率
        self.margins = self.softmax[self.y, range(N)]

        data_loss = np.sum(- np.log(self.margins)) / N
        regular_loss = self.lam * np.sum(np.square(W))
        loss = data_loss + regular_loss

        return loss


    def backward(self, step_size, prop_grad = 1.0):

        N = self.y.shape[0]

        s = np.zeros(self.softmax.shape)
        s[self.y, range(N)] = self.margins[range(N)]  # 10 x 50000
        dw = s.dot(self.X)

        # dw = np.zeros(self.W.shape)  # 10 x 3073
        # for i in range(self.y.shape[0]):
        #     dw[self.y[i]] += (self.margins[i] - 1) * self.X[i]

        dw = dw / N

        # 计算 regular_loss 的积分
        dr = 2 * self.lam * self.W

        self.grad = dw + dr
        # 更新本层权重
        self.W += - step_size * self.grad

        prop_grad = prop_grad * self.grad

        return prop_grad


if __name__ == '__main__':
    from load_data import load_CIFAR10, sample_training_data

    X_train, Y_train = load_CIFAR10('../../data/cifar10/')[0:2]
    X_train = np.append(X_train, np.ones((X_train.shape[0], 1)), axis = 1)
    data_train = [X_train, Y_train]
    data_batch = sample_training_data(data_train, 256)  # 256个数据
    W = np.random.rand(10, 3073) * 0.001  # 随机权重向量
    # W = np.zeros((10, 3073))
    softmax = SoftmaxLayer(W = W, X = data_batch[0], y = data_batch[1])
    loss_original = softmax.forward()
    print('original loss: %f' % (loss_original,))
    min_loss = loss_original
    for step_size_log in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]:
        step_size = 10 ** step_size_log

        softmax.forward(weights = W)  # 重置 W
        softmax.backward(step_size = step_size)  # 更新 W
        loss_new = softmax.forward()  # 计算 loss
        print('for step size %.10f new loss: %.7f' % (step_size, loss_new))
        if loss_new < min_loss:
            min_loss = loss_new
            best_step_size = step_size

    print("best step size %.10f" % (best_step_size,))

    softmax.forward(weights = W)
    i = 0
    while True:
        data_batch = sample_training_data(data_train, 256)  # 256个数据

        loss_before = softmax.forward(data = data_batch)
        softmax.backward(step_size = best_step_size)
        loss_after = softmax.forward(data = data_batch)
        i += 1
        print("loop %d loss: %f" % (i, loss_after))
        if loss_after < 10 ** -5:
            break
