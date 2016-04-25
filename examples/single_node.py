#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2016-02-12 16:46:25
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-04-24 12:10:09
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.


import numpy as np

from utils import load_digits

from peregrine import descend


class LogregExecutor(object):
    """docstring for LogregExecutor"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.n_samples = self.features.shape[0]
        self.n_features = self.features.shape[1]

    @property
    def n_variables(self):
        return self.n_features

    def eval_obj(self, w):
        w = np.array(w, copy=False)
        X, Y = self.features, self.labels
        self.e_ywx = np.exp(Y*X.dot(w))
        return np.average(np.log1p(1/self.e_ywx))

    def eval_grad(self, w, df):
        w = np.array(w, copy=False)
        df = np.array(df, copy=False)
        X, Y, n = self.features, self.labels, self.n_samples
        a = -Y/(1+self.e_ywx)/n
        (X*a[:,np.newaxis]).sum(axis=0, out=df)
prob = LogregExecutor(*load_digits())



# from peregrine.objectives import Worker
# from peregrine.objectives import Executor
# def logreg_from(w, worker):
#     """ Log Reg Single worker (dense or sparse). """
#     X, Y, n = worker.features, worker.labels, worker.n_samples
#     e_ywx = np.exp(Y*X.dot(w))
#     loss = np.average(np.log1p(1/e_ywx))
#     a = -Y/(1+e_ywx)/n
#     df = X.transpose().dot(a)
#     return loss, df

# def collect_from(worker, w, func):
#     return func(w, worker)

# single_worker = Worker(*load_digits())
# prob = Executor(single_worker,
#                 single_worker.n_samples,
#                 single_worker.n_features,
#                 merge_func=collect_from,
#                 transition_func=logreg_from,
#                 cached=True, l2_reg=0.1)

initial_model = np.random.random(prob.n_variables)
# initial_model = []
descend(prob, list(initial_model), initial_stepsize=0.1,
        verbose=1, max_iter=30, l1_reg=0.002, precision='f')

