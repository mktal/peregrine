#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2016-02-12 16:02:25
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-07-05 14:04:07
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.

import numpy as np
from scipy.sparse import csr_matrix
from itertools import repeat
from itertools import chain


def _make_matrix_dense(rows):
    Y = np.array([r.label for r in rows])
    X = np.array([r.features.toArray() for r in rows])
    return X, Y


def _make_matrix_sparse(rows):
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
    return X, Y


def collect(datasets, w, func):
    def seqOp(v, row):
        l, g = func(w, row)
        return v[0]+l, v[1]+g, row.n_samples

    def combOp(v1, v2):
        n1, n2 = v1[2], v2[2]
        total = n1 + n2
        f = (n1*v1[0] + n2*v2[0]) / total
        g = (n1*v1[1] + n2*v2[1]) / total
        return f, g, total

    res = datasets.treeAggregate((0, 0), seqOp, combOp)
    return res[0], res[1]


class Worker(object):
    def __init__(self, features, labels):
        assert(isinstance(features, (np.ndarray, csr_matrix)))
        assert(isinstance(labels, np.ndarray))
        self.features = features
        self.labels = labels
        self.n_samples = self.features.shape[0]
        self.n_features = self.features.shape[1]

    @classmethod
    def from_rows(cls, rows, dense=False):
        if not dense: return cls(*_make_matrix_sparse(rows))
        else: return cls(*_make_matrix_dense(rows))


class Executor(object):
    def __init__(self, labeledPoints, n_samples, n_features, merge_func,
                 transition_func, cached=False, l2_reg=0.):
        self.labeledPoints = labeledPoints
        self.n = n_samples
        self.d = n_features
        self.cached = cached
        self.l2_reg = l2_reg
        self._transition_func = transition_func
        self._merge_func = merge_func
        self._cached_grad = None

    @property
    def n_variables(self):
        return self.d

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
