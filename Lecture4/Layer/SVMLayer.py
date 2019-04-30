#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   CS231n-Note 

File Name:  SVMLayer.py

"""
import numpy as np

__author__ = 'Welkin'
__date__ = '2017/8/12 13:31'


class SVMLayer(object):
    def __init__(self, W, X, y, delta = 1.0, lam = 0.1):
        """
        :param W: 权重。对于 CIFAR-10, W 应该是 10 x 3073
        :param X: 输入（图片）数据。如果直接读取 CIFAR-10 数据，X 为 50000 x 3073, 在和 W 做运算时应该转置
        :param y: 每张图片对应的标签。CIFAR-10 中 为 50000 x 1
        :param delta: SVM 的边缘。默认为1.0
        :param lam: 正则化损失所占比例，超参数，默认为0.1
        """
        self.W = np.copy(W)
        self.X = np.copy(X)
        self.y = np.copy(y)
        self.delta = delta
        self.lam = lam


    def forward(self, data = None, weights = None, delta = None, lam = None):
        """
        前向传播：分别计算 SVM 和正则化的损失（loss），然后相加到一起，返回
        :param data: data 是用于前向计算和反向求积分的 (X, y) 数据集，如果为 None 则表示继续使用上一次的数据进行计算
        :param W:
        :param delta:
        :param lam:
        :return:
        """
        if weights is not None:
            self.W = np.copy(weights)
        if data is not None:
            self.X, self.y = data
        if delta is not None:
            self.delta = delta
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
        self.scores = W.dot(X)  # e.g. 10 x 50000
        svm = self.scores - self.scores[self.y, range(N)] + self.delta  # e.g. 10 x 50000

        # ReLU 激活函数
        self.margins = np.maximum(0, svm)
        self.margins[self.y, range(N)] = 0
        # SVM 层在神经网络的中应该输出 margins 的数据给下一层，或者作为输出层的最终结果
        # 这里是为了使优化权重的过程更容易理解，所以输出了 loss 数据，在神经网络的应用中可以不计算 loss

        data_loss = np.sum(self.margins) / N
        regular_loss = self.lam * np.sum(np.square(W))
        loss = data_loss + regular_loss

        return loss


    def backward(self, step_size, prop_grad = 1.0):
        """
        :param prop_grad: 由计算图中下一层传来的梯度，应该和权重 W 有相同维度，即 10 x 3073
        :return:
        """
        N = self.y.shape[0]
        # dmargins = np.ones(self.margins.shape)
        indicator = np.copy(self.margins)  # 10 x 50000
        indicator[indicator > 0] = 1

        indicator[self.y, range(N)] = - np.sum(indicator[:, range(N)], axis = 0)
        # dsvm = indicator * dmargins
        # dscores = 1.0 * dsvm
        # dw = dscores x X.T

        # 计算 data_loss 的积分
        # 计算第 1 个 x 的对 W 的积分，10 x 3073 = (10 x 1 * 1 x 3073) * 50000 = 10 x 50000 * 50000 x 3073
        dw = indicator.dot(self.X) / N

        # 计算 regular_loss 的积分
        dr = 2 * self.lam * self.W

        self.grad = dw + dr
        # 更新本层权重
        self.W += - step_size * self.grad

        prop_grad = prop_grad * self.grad  # 将此层的梯度与传来的梯度相乘，然后继续向上一层传播
        return prop_grad


if __name__ == '__main__':
    from load_data import load_CIFAR10, sample_training_data

    X_train, Y_train = load_CIFAR10('../../data/cifar10/')[0:2]
    X_train = np.append(X_train, np.ones((X_train.shape[0], 1)), axis = 1)
    data_train = [X_train, Y_train]
    data_batch = sample_training_data(data_train, 256)  # 256个数据
    W = np.random.rand(10, 3073) * 0.001  # 随机权重向量
    # W = np.zeros((10, 3073))
    svm = SVMLayer(W = W, X = data_batch[0], y = data_batch[1])
    loss_original = svm.forward()
    print('original loss: %f' % (loss_original,))
    min_loss = loss_original
    for step_size_log in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]:
        step_size = 10 ** step_size_log

        svm.forward(weights = W)  # 重置 W
        svm.backward(step_size = step_size)  # 更新 W
        loss_new = svm.forward()  # 计算 loss
        print('for step size %.10f new loss: %.7f' % (step_size, loss_new))
        if loss_new < min_loss:
            min_loss = loss_new
            best_step_size = step_size

    print("best step size %.10f" % (best_step_size,))

    svm.forward(weights = W)
    i = 0
    while True:
        data_batch = sample_training_data(data_train, 256)  # 256个数据

        loss_before = svm.forward(data = data_batch)
        svm.backward(step_size = best_step_size)
        loss_after = svm.forward(data = data_batch)
        i += 1
        print("loop %d loss: %f" % (i, loss_after))
        if loss_after < 10 ** -5:
            break
