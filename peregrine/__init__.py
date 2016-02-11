#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2016-01-04 20:21:11
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-02-11 00:04:45
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.


__all__ = ['descend', 'DistributedWithSpark']

from ._peregrine import _train

import numpy as np
from scipy.sparse import csr_matrix
from collections import namedtuple
from itertools import repeat
from itertools import chain


class DistributedWithSpark(object):
    """ Distributed Matrix form"""
    def __init__(self, labeledPoints, transition_func,
                 cached=False, l2_reg=0., dense=False):
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

            res = datasets.treeAggregate((0, 0), seqOp, combOp)
            assert(res[2] == self.n)
            return res[0], res[1]

        self.labeledPoints = labeledPoints
        self.n = self.labeledPoints.count()
        self.d = self.labeledPoints.first().features.size
        self.cached = cached
        self.l2_reg = l2_reg
        self._transition_func = transition_func
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


def descend(objective, verbose=2, opt_tol=1e-8, max_iter=500,
            memory=10, l1_reg=1e-6, precision='f', **kwargs):
    """Proximal quasi-Newton (L-BFGS) minimizer for composite function

    Parameters
    ----------

    objective : the objective function to be minimized. Required to implement
        two functions eval_obj (function value) and eval_obj (gradient value)

    dim : int, dimension of the objective function

    verbose : int, optional (default=2)
        set verbosity of the output. Set to 0 for silence

    opt_tol : float, optional (default=1e-8)
        set tolerance of termination criterion
        ``final ista step size <= eps*(initial ista step size)``

    max_iter : int, optional (default=500)
        max number of iterations

    memory : int, optional (default=10)
        limited memory of L-BFGS. Use cache size to the order of
        ``2*memory*dim*sizeof(precision)``
        to approximate the curvature of objective for faster convergence

    l1_reg : float, optional (default=1e-6)

    """
    f, g, dim = objective.eval_obj, objective.eval_grad, objective.shape[1]
    return _train(f, g, dim, verbose, opt_tol,
                  max_iter, memory, l1_reg, precision)
