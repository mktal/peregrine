#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2015-12-17 15:01:49
# @Last Modified by:   xtang
# @Last Modified time: 2016-01-06 22:54:39
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.


import numpy as np
from sklearn.datasets import load_svmlight_file
from ..peregrine import descend


class LogReg(object):
    """docstring for LogReg"""
    def __init__(self, X, Y, cached=False):
        self.X = X
        self.Y = Y
        self.n, self.d = self.shape
        self.cached = cached

    @property
    def shape(self):
        return self.X.shape

    def eval_obj(self, w):
        w = np.array(w, copy=False)
        e_ywx_inv = np.exp(-self.Y*np.dot(self.X, w))
        loss = np.average(np.log1p(e_ywx_inv))
        if self.cached:
            self._cache = e_ywx_inv
        return loss

    def eval_grad(self, w, df):
        w = np.array(w, copy=False)
        df = np.array(df, copy=False)
        if self.cached:
            e_ywx = 1/self._cache
        else:
            e_ywx = np.exp(self.Y*np.dot(self.X, w))
        a = -self.Y/(1+e_ywx)/self.n
        # np.dot(a, self.X, out=df)
        np.copyto(df, np.dot(a, self.X))

data, target = load_svmlight_file('/Users/xtang/data/mnist_0_6')
data = np.asarray(data.todense())
prob = LogReg(data, target, cached=True)
descend(prob, verbose=1, precision='d')
