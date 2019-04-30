#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   CS231n-Note 

File Name:  layer.py

"""
import numpy as np

__author__ = 'Welkin'
__date__ = '2017/11/21 18:25'


class layer(object):
    """
    神经网络每一层的节点集合
    """

    def __init__(self):
        self.W = None
        self.X = None
        self.b = None
        self.output = None
        pass

    def forward(self, X_data):
        pass

    def backward(self, dL, learn_rate):
        pass

    def gradient(self, dL):
        pass


class input_layer(layer):
    """
    输入层
    """

    def __init__(self, number_this):
        super().__init__()

    def forward(self, X_data):
        self.X = X_data
        self.output = self.X
        return self.output

    def gradient(self, dL):
        dX = dL
        return dX


class hidden_layer(layer):
    """
    隐藏层
    """

    def __init__(self, number_before, number_this, f = None, df = None, reg_lam = 0):
        """
        用上一层节点数和本层节点数来初始化隐藏层
        """
        super().__init__()
        self.number_before = number_before
        self.number_this = number_this
        self.W = np.zeros((number_this, number_before))
        self.b = np.zeros((number_this, 1))
        if f is not None and df is not None:
            self.f = f
            self.df = df
        self.reg_lam = reg_lam

    def forward(self, X_data):
        self.X = X_data
        self.mid_value = np.dot(self.W, self.X) + self.b
        self.output = self.f(self.mid_value)
        return self.output

    def backward(self, dL, learn_rate):
        dW, db, dX = self.gradient(dL)
        self.W += - learn_rate * dW
        self.b += - learn_rate * db

        return dX

    def gradient(self, dL):
        dmid_value = self.df(self.mid_value, self.output, dL)
        self._dW = np.dot(dmid_value, self.X.T)
        self._dW += self.dreg()
        self._db = np.sum(dmid_value, axis = 1).reshape((-1, 1))

        dX = np.dot(self.W.T, dmid_value)
        return self._dW, self._db, dX

    def f(self, x):
        """激活函数"""
        return x

    def df(self, x, y, dL):
        return 1

    def reg_loss(self):
        """正则化损失"""
        return 0

    def dreg(self):
        return 0


class output_layer(hidden_layer):
    pass


class L2_hidden_layer(hidden_layer):
    def reg_loss(self):
        return 0.5 * self.reg_lam * np.sum(self.W ** 2)

    def dreg(self):
        return self.reg_lam * self.W
