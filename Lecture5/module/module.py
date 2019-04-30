#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   CS231n-Note 

File Name:  module.py

"""
import numpy as np

__author__ = 'Welkin'
__date__ = '2017/8/14 17:51'


class module(object):
    def __init__(self, name, in_num, out_num, x, step_size = None, lam = 0.01,
                 active_fun = lambda x: x,
                 dactive_fun = lambda i, o: np.ones(o.shape)):
        self.name = name
        self.in_num = in_num
        self.out_num = out_num
        # self.W = np.random.randn(out_num, in_num) / np.sqrt(in_num)  # 在 tanh 中效果好，在 ReLU 中方差下降的更快
        self.W = np.random.randn(out_num, in_num) / np.sqrt(in_num / 2.0)  # 不除以2.0，每层输出会以指数级收缩
        self.b = np.zeros((out_num, 1))
        self.X = x
        self.step_size = step_size
        self.lam = lam
        self.active_fun = active_fun
        self.dactive_fun = dactive_fun

        # self.output = np.zeros((out_num, x.shape[1]))
        self.output = self.forward()


    # @property
    # def output(self):
    #     return self.forward()

    @property
    def L1(self):
        return (np.sum(np.abs(self.W)))


    @property
    def L2(self):
        return (np.sum(self.W ** 2))


    def forward(self):
        self.input = self.W.dot(self.X) + self.b
        # active_fun 激活函数为一元函数，为本层输入的数据和输出的数据做一一映射，即 D x N 到 D x train_num 的一一映射
        self.output = self.active_fun(self.input)
        return self.output


    def backward(self, output_grad = 1.0):
        # 下层传来的梯度维数应该和本层输出数据的维度相同，即 D x N 维矩阵，应该先转换成 D x 1，再和激活函数的导数对应项一一相乘
        dinput = self.dactive_fun(self.input, self.output) * output_grad
        # 本层权重的梯度，用于更新本层权重
        self.dw = dinput.dot(self.X.T)
        self.dw += self.lam * 2 * self.W
        # 从上一层接收到的数据的梯度，用于反向传播
        self.dx = self.W.T.dot(dinput)
        self.db = np.sum(dinput * 1.0, axis = 1).reshape((-1, 1))

        # 更新权重
        self.W += - self.step_size * self.dw
        self.b += - self.step_size * self.db
        return self.dx


if __name__ == '__main__':

    m = module(1, 2, 3, 4)

    print(m.forward)
