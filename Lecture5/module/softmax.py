#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   CS231n-Note 

File Name:  softmax.py

"""
import numpy as np
from Lecture5.module.module import module

__author__ = 'Welkin'
__date__ = '2017/8/16 02:58'


class Softmax(module):
    def __init__(self, name, in_num, out_num, x, step_size, **lam):
        super().__init__(name, in_num, out_num, x, step_size, **lam, active_fun = self.softmax,
                         dactive_fun = self.dsoftmax)


    def forward(self):
        return super().forward()


    def backward(self, output_grad):
        return super().backward(output_grad)


    def softmax(self, x):
        scores = x - np.max(x, axis = 0)
        return np.exp(scores) / np.sum(np.exp(scores), axis = 0)


    def dsoftmax(self, x, y):
        return y * (1 - y)
