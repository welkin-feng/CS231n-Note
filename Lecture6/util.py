#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   CS231n-Note 

File Name:  util.py

"""
import numpy as np

__author__ = 'Welkin'
__date__ = '2017/11/21 19:31'


def relu(x):
    input = np.copy(x)
    input[input < 0] = 0
    return input


def drelu(x, y, dL):
    dx = np.copy(y)
    dx[dx > 0] = 1
    return dL * dx


def softmax(x):
    scores = x - np.max(x, axis = 0)
    return np.exp(scores) / np.sum(np.exp(scores), axis = 0)


def dsoftmax(x, y, dL):
    # broadcast = y[:, np.newaxis] * np.ones(y.shape)
    return -np.sum(dL * y * y[:, np.newaxis], axis = 1) + dL * y
