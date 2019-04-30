#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   CS231n-Note 

File Name:  ReLU.py

"""
import numpy as np
from Lecture5.module.module import module

__author__ = 'Welkin'
__date__ = '2017/8/14 17:49'


class ReLU(module):
    def __init__(self, name, in_num, out_num, x, step_size, **lam):
        super().__init__(name, in_num, out_num, x, step_size, **lam, active_fun = self.relu, dactive_fun = self.drelu)


    def forward(self):
        return super().forward()


    def backward(self, output_grad):
        return super().backward(output_grad)


    def relu(self, x):
        input = np.copy(x)
        input[input < 0] = 0
        return input


    def drelu(self, x, y):
        dx = np.copy(y)
        dx[dx > 0] = 1
        return dx
