#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2015-12-17 14:53:14
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2015-12-21 01:10:48
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.

import numpy as np
from pprint import pprint
from collections import namedtuple


def sLogReg(lmd=0.0001):
    """ Log Reg Single row"""
    def obj_and_grad(w, row):
        x, y = row.features, row.label
        e_ywx = np.exp(y*x.dot(w))
        # loss = np.log(1/e_ywx + 1)
        loss = np.log1p(1/e_ywx)
        a = -1*y/(1+e_ywx)
        reg = lmd*0.5*w.dot(w)
        df = lmd*w
        df[x.indices] += a*x.values
        return loss + reg, df
    return obj_and_grad

def sLogRegD(lmd=0.0001):
    """ Log Reg Single row Dense"""
    def obj_and_grad(w, row):
        X, Y, n = row.features, row.labels, row.n_samples
        e_ywx = np.exp(Y*np.dot(X, w))
        loss = np.average(np.log1p(1/e_ywx))
        a = -Y/(1+e_ywx)/n
        df = np.dot(a, X)
        return loss + lmd*0.5*w.dot(w), df
    return obj_and_grad


class LogRegD(object):
    """ LogReg Distributed"""
    def __init__(self, labeledPoints, cached=False, l2_reg=0.):
        self.labeledPoints = labeledPoints
        self.n = self.labeledPoints.count()
        self.d = self.labeledPoints.first().features.size
        self.cached = cached
        # self.labeledPoints = self.labeledPoints.glom()
        self._single_eval = sLogReg(l2_reg)
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


class LogRegDD(LogRegD):
    """ LogReg Distributed Dense"""
    def __init__(self, labeledPoints, cached=False, l2_reg=0.):
        def make_matrix(rows):
            _Row = namedtuple('Row', 'features, labels, n_samples')
            Y = np.array([r.label for r in rows])
            X = np.array([r.features.toArray() for r in rows])
            return _Row(X, Y, X.shape[0])

        super(LogRegDD, self).__init__(labeledPoints,
                                       cached=cached,
                                       l2_reg=l2_reg)
        self.labeledPoints = labeledPoints.glom().map(make_matrix).cache()
        # import ipdb; ipdb.set_trace()
        self._single_eval = sLogRegD(l2_reg)

    def _full_batch(self, w, func):
        # row[1] is a list of LabeledData
        def seqOp(v, row):
            l, g = func(w, row)
            return v[0]+l, v[1]+g, row.n_samples
        def combOp(v1, v2):
            n1, n2 = v1[2], v2[2]
            total = n1 + n2
            f = (n1*v1[0] + n2*v2[0]) / total
            g = (n1*v1[1] + n2*v2[1]) / total
            return f, g, total
        res = self.labeledPoints.treeAggregate((0,0), seqOp, combOp)
        assert(res[2] == self.n)
        return res[0], res[1]


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
