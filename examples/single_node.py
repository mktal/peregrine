#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2016-02-12 16:46:25
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-02-12 23:24:05
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.

from utils import load_digits
from peregrine import descend
from peregrine.objectives import logistic_regression
from peregrine.objectives import Worker


dataset = Worker(*load_digits())
prob = logistic_regression(dataset, l2_reg=0.001)
descend(prob, verbose=1, max_iter=60, l1_reg=0.001, precision='f')

prob = logistic_regression(dataset, l2_reg=0.001, tensorflow=True)
descend(prob, verbose=1, max_iter=60, l1_reg=0.001, precision='f')
