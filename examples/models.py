#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2015-12-17 14:53:14
# @Last Modified by:   xtang
# @Last Modified time: 2016-01-04 12:19:55
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix
from pprint import pprint
from collections import namedtuple
from functools import partial
from itertools import izip
from itertools import imap
from itertools import repeat
from itertools import chain


class BaseLogReg(object):
    """Base class for log reg"""
    def __init__(self, labeledPoints,
                 transition_func, merge_func,
                 cached=False, l2_reg=0.):
        self.labeledPoints = labeledPoints
        self.n = self.labeledPoints.count()
        self.d = self.labeledPoints.first().features.size
        self.cached = cached
        self.l2_reg = l2_reg
        self._transition_func = transition_func
        self._merge_func = merge_func
        self._cached_grad = None

    @property
    def shape(self):
        return self.n, self.d

    @property
    def eval_obj(self):
        if self.cached:
            return self._eval_obj_c
        else:
            return self._eval_obj

    @property
    def eval_grad(self):
        if self.cached:
            return self._eval_grad_c
        else:
            return self._eval_grad

    def _eval_obj(self, w):
        w = np.array(w, copy=False)
        l, _ = self._merge_func(self.labeledPoints, w, self._transition_func)
        l += self.l2_reg*0.5*w.dot(w)
        return l

    def _eval_obj_c(self, w):
        w = np.array(w, copy=False)
        l, self._cached_grad = self._merge_func(self.labeledPoints,
                                                w, self._transition_func)
        l += self.l2_reg*0.5*w.dot(w)
        self._cached_grad += self.l2_reg*w
        return l

    def _eval_grad(self, w, df):
        w = np.array(w, copy=False)
        df = np.array(df, copy=False)
        _, self._cached_grad = self._merge_func(self.labeledPoints,
                                                w, self._transition_func)
        self._cached_grad += self.l2_reg*w
        np.copyto(df, self._cached_grad)

    def _eval_grad_c(self, w, df):
        df = np.array(df, copy=False)
        np.copyto(df, self._cached_grad)


class LogRegDV(BaseLogReg):
    """ LogReg Distributed Vector form"""
    def __init__(self, labeledPoints, cached=False, l2_reg=0.):
        def _merge(datasets, w, func):
            def seqOp(v, rows):
                agg = lambda r, v: (r[0]+v[0], r[1]+v[1])
                l, g = reduce(agg, (func(w, row) for row in rows))
                return v[0]+l, v[1]+g
            def combOp(v1, v2):
                return v1[0]+v2[0], v1[1]+v2[1]
            total = datasets.treeAggregate((0,0), seqOp, combOp)
            return total[0] / self.n, total[1] / self.n

        def _transition(w, row):
            """ Log Reg Single row"""
            x, y = row.features, row.label
            e_ywx = np.exp(y*x.dot(w))
            loss = np.log1p(1/e_ywx)
            a = -1*y/(1+e_ywx)
            df = np.zeros_like(w)
            df[x.indices] += a*x.values
            return loss, df

        super(LogRegDV, self).__init__(labeledPoints, _transition, _merge,
                                      cached=cached, l2_reg=l2_reg)
        self.labeledPoints = self.labeledPoints.glom().cache()


class LogRegDM(BaseLogReg):
    """ LogReg Distributed Matrix form"""
    def __init__(self, labeledPoints, cached=False, l2_reg=0., dense=False):
        def make_matrix_dense(rows):
            _Row = namedtuple('Row', 'features, labels, n_samples')
            Y = np.array([r.label for r in rows])
            X = np.array([r.features.toArray() for r in rows])
            return _Row(X, Y, X.shape[0])

        def make_matrix_sparse(rows):
            _Row = namedtuple('Row', 'features, labels, n_samples')
            Y = np.array([r.label for r in rows])
            n, d = len(rows), rows[0].features.size
            col = np.array(list(chain.from_iterable(
                           r.features.indices for r in rows)))
            row = np.array(list(chain.from_iterable(
                           repeat(i, len(r.features.indices))
                           for i, r in enumerate(rows))))
            vals = np.array(list(chain.from_iterable(
                            r.features.values for r in rows)))
            X = csr_matrix((vals, (row, col)), shape=(n, d))
            return _Row(X, Y, n)

        def _merge(datasets, w, func):
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
            res =datasets.treeAggregate((0,0), seqOp, combOp)
            assert(res[2] == self.n)
            return res[0], res[1]

        def _transition(w, row):
            """ Log Reg Single row (dense or sparse). """
            X, Y, n = row.features, row.labels, row.n_samples
            e_ywx = np.exp(Y*X.dot(w))
            loss = np.average(np.log1p(1/e_ywx))
            a = -Y/(1+e_ywx)/n
            # df = np.dot(a, X)
            df = X.transpose().dot(a)
            return loss, df

        super(LogRegDM, self).__init__(labeledPoints, _transition, _merge,
                                      cached=cached, l2_reg=l2_reg)
        make_matrix = make_matrix_dense if dense else make_matrix_sparse
        self.labeledPoints = self.labeledPoints.glom().map(make_matrix).cache()


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
