#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2015-12-17 14:53:14
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2015-12-17 23:38:47
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.

import numpy as np
from pprint import pprint


def LogRegS(lmd=0.0001):
    def obj_and_grad(w, row):
        x, y = row.features, row.label
        e_ywx = np.exp(y*x.dot(w))
        loss = np.log(1/e_ywx + 1)
        a = -1*y/(1+e_ywx)
        reg = lmd*0.5*w.dot(w)
        df = lmd*w
        df[x.indices] += a*x.values
        return loss + reg, df
    return obj_and_grad


class LogRegD(object):
    """docstring for LogReg"""
    def __init__(self, labeledPoints, cached=False, l2_reg=0.):
        self.labeledPoints = labeledPoints
        self.n = self.labeledPoints.count()
        self.d = self.labeledPoints.first().features.size
        self.cached = cached
        # self.labeledPoints = self.labeledPoints.glom()
        self._single_eval = LogRegS(l2_reg)
        self._cached_grad = None

    @property
    def shape(self):
        return self.n, self.d

    def _full_batch(self, w, func):
        # row[1] is a list of LabeledData
        def seqOp(v, row):
            l, g = func(w, row)
            return v[0]+l, v[1]+g
        def combOp(v1, v2):
            return v1[0]+v2[0], v1[1]+v2[1]
        total = self.labeledPoints.treeAggregate((0,0), seqOp, combOp)
        return total[0] / self.n, total[1] / self.n

    def eval_obj(self, w):
        w = np.array(w, copy=False)
        l, self._cached_grad = self._full_batch(w, self._single_eval)
        if not self.cached:
            self._cached_grad = None
        return l

    def eval_grad(self, w, df):
        w = np.array(w, copy=False)
        df = np.array(df, copy=False)
        if self._cached_grad is None:
            _, self._cached_grad = self._full_batch(w, self._single_eval)
        np.copyto(df, self._cached_grad)


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
