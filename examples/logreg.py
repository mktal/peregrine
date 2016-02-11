#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2016-01-05 21:51:20
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-02-10 23:05:28
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.

from pyspark import SparkConf, SparkContext
from pyspark.mllib.util import MLUtils
from utils import SparkController
from ..peregrine import descend
import numpy as np
from scipy.sparse import csr_matrix
from collections import namedtuple
from functools import partial
from itertools import izip
from itertools import imap
from itertools import repeat
from itertools import chain


class LogRegDM(object):
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
            res =datasets.treeAggregate((0, 0), seqOp, combOp)
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

        self.labeledPoints = labeledPoints
        self.n = self.labeledPoints.count()
        self.d = self.labeledPoints.first().features.size
        self.cached = cached
        self.l2_reg = l2_reg
        self._transition_func = _transition
        self._merge_func = _merge
        self._cached_grad = None
        make_matrix = make_matrix_dense if dense else make_matrix_sparse
        self.labeledPoints = self.labeledPoints.glom().map(make_matrix).cache()

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


def peregrineLogReg(sc, data_path):
    dataset = MLUtils.loadLibSVMFile(sc, data_path, minPartitions=8).cache()
    prob = LogRegDM(dataset, cached=True, l2_reg=0.001)
    descend(prob, verbose=1, max_iter=30, l1_reg=0.001, precision='f')


if __name__ == '__main__':
    with SparkController() as sc:
        peregrineLogReg(sc, '/Users/xtang/Documents/cahow/examples/data/a9a')

