#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2016-02-12 16:46:25
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-04-21 23:46:47
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.


import numpy as np

from utils import load_digits

from peregrine import descend
from peregrine.objectives import Worker
from peregrine.objectives import Executor


def logreg_from(w, worker):
    """ Log Reg Single worker (dense or sparse). """
    X, Y, n = worker.features, worker.labels, worker.n_samples
    e_ywx = np.exp(Y*X.dot(w))
    loss = np.average(np.log1p(1/e_ywx))
    a = -Y/(1+e_ywx)/n
    df = X.transpose().dot(a)
    return loss, df


def collect_from(worker, w, func):
    return func(w, worker)


single_worker = Worker(*load_digits())
prob = Executor(single_worker,
                single_worker.n_samples,
                single_worker.n_features,
                merge_func=collect_from,
                transition_func=logreg_from,
                cached=True, l2_reg=0.1)
descend(prob, verbose=1, max_iter=30, l1_reg=0.002, precision='f')

