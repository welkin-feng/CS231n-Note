#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   CS231n-Note 

File Name:  Optimization.py

"""
import numpy as np

__author__ = 'Welkin'
__date__ = '2017/8/12 16:25'


class Optimization(object):
    def eval_numerical_gradient(loss_fun, data, weights):
        """
        一个f在x处的数值梯度法的简单实现
        - f 是只有一个参数的函数
        - x 是计算梯度的点
        """

        fw = loss_fun(data, weights)  # 在原点计算函数值
        grad = np.zeros(weights.shape)
        h = 0.00001

        # 对x中所有的索引进行迭代
        it = np.nditer(weights, flags = ['multi_index'], op_flags = ['readwrite'])
        while not it.finished:

            # 计算w+h处的函数值
            iw = it.multi_index
            old_value = weights[iw]
            weights[iw] = old_value + h  # 增加h
            fwh = loss_fun(data, weights)  # 计算f(w + h)
            weights[iw] = old_value  # 存到前一个值中 (非常重要)

            # 计算偏导数
            grad[iw] = (fwh - fw) / h  # 坡度
            it.iternext()  # 到下个维度

        return grad, fw


# 要使用上面的代码我们需要一个只有一个参数的函数
# (在这里参数就是权重)所以也包含了X_train和Y_train
def CIFAR10_loss_fun(data, weights):
    """
    data = [X_train, Y_train]
    """
    from Lecture3.Loss import L_SVM
    return L_SVM(data[0], data[1], weights)


if __name__ == '__main__':
    from load_data import load_CIFAR10, sample_training_data

    X_train, Y_train = load_CIFAR10('../data/cifar10/')[0:2]
    X_train = np.append(X_train, np.ones((X_train.shape[0], 1)), axis = 1)
    data_train = [X_train, Y_train]
    data_batch = sample_training_data(data_train, 256)  # 256个数据
    W = np.random.rand(10, 3073) * 0.001  # 随机权重向量
    op = Optimization()
    df, loss_original = op.eval_numerical_gradient(CIFAR10_loss_fun, data_batch, W)  # 得到梯度、初始损失值

    print('original loss: %f' % (loss_original,))
    min_loss = loss_original
    # 查看不同步长的效果
    for step_size_log in [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1]:
        step_size = 10 ** step_size_log

        W_new = W - step_size * df  # 权重空间中的新位置
        loss_new = CIFAR10_loss_fun(data_batch, W_new)
        print('for step size %f new loss: %.7f' % (step_size, loss_new))
        if loss_new < min_loss:
            min_loss = loss_new
            best_step_size = step_size

    print("best step size %.7f" % (best_step_size,))

    for i in range(300):
        data_batch = sample_training_data(data_train, 256)  # 256个数据
        weights_grad, loss = op.eval_numerical_gradient(CIFAR10_loss_fun, data_batch, W)
        print("loop %d loss: %f" % (i, loss))
        if loss < 10 ** -5:
            break
        W += - best_step_size * weights_grad  # 参数更新

    pass
    # for 循环输出：
    # loop 0 loss: 17.405877
    # loop 1 loss: 10.058977
    # loop 2 loss: 11.290340
    # loop 3 loss: 11.520325
    # loop 4 loss: 10.720532
    # loop 5 loss: 10.440468
    # loop 6 loss: 9.661210
    # loop 7 loss: 10.046043
    # loop 8 loss: 11.326161
    # loop 9 loss: 10.621673
    # loop 10 loss: 10.449573
    # loop 11 loss: 10.644136
    # loop 12 loss: 10.531986
    # loop 13 loss: 9.246973
    # loop 14 loss: 10.797842
