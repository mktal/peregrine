#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2016-03-08 21:48:34
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-03-09 16:05:17
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.

from peregrine import descend
from peregrine.objectives import logistic_regression
from peregrine.objectives import Worker

from sklearn.datasets import load_svmlight_file
from sklearn import datasets


def load_digits():
    train = datasets.load_digits()
    data, target = train.data, train.target.copy()
    target[target != 0] = -1
    target[target == 0] = 1
    print('load digits for classification: {}'.format(data.shape))
    return data, target

dataset = Worker(*load_digits())
prob = logistic_regression(dataset, l2_reg=0.001)
descend(prob, verbose=1, max_iter=60, l1_reg=0.001, precision='d')
