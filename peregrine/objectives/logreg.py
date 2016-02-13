#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2016-02-12 16:00:32
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-02-12 23:59:55
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.

from functools import partial

import numpy as np

from .execution import Worker
from .execution import Executor
from .execution import single_merge
from .execution import weighted_merge


def logistic_regression(labeledPoints, l2_reg=0.,
                        dense=False, tensorflow=False):
    if isinstance(labeledPoints, Worker):
        n, d = labeledPoints.n_samples, labeledPoints.n_features
        _trans_func = LogRegWithTF(d).run if tensorflow else logreg_local
        return Executor(labeledPoints, n, d, single_merge, _trans_func,
                        cached=True, l2_reg=l2_reg)
    else:
        from pyspark import RDD
        if not isinstance(labeledPoints, RDD):
            raise ValueError('logistic regression: labeledPoints wrong type!')
        n = labeledPoints.count()
        d = labeledPoints.first().features.size
        dense = True if tensorflow else dense
        make_matrix = partial(Worker.from_rows, dense=dense)
        labeledPoints = labeledPoints.glom().map(make_matrix).cache()
        _trans_func = LogRegWithTF(d).run if tensorflow else logreg_local
        return Executor(labeledPoints, n, d, weighted_merge, _trans_func,
                        cached=True, l2_reg=l2_reg)


# transition function
def logreg_local(w, worker):
    """ Log Reg Single worker (dense or sparse). """
    X, Y, n = worker.features, worker.labels, worker.n_samples
    e_ywx = np.exp(Y*X.dot(w))
    loss = np.average(np.log1p(1/e_ywx))
    a = -Y/(1+e_ywx)/n
    df = X.transpose().dot(a)
    return loss, df


class LogRegWithTF(object):
    """logistic regression using tensorflow"""
    def __init__(self, dim):
        self.dim = dim
        self.sess = None

    def _initialize(self, dim):
        import tensorflow as tf
        self.dim = dim
        self.w = tf.Variable(tf.zeros([dim]), name='model')
        self.x = tf.placeholder("float", [None, dim], name='features')
        self.y = tf.placeholder("float", [None], name='target')
        self.loss = -tf.reduce_sum(tf.log(tf.sigmoid(
                    self.y*tf.reduce_sum(self.x*self.w, 1))))
        self.grads = tf.gradients(self.loss, self.w)[0]
        self.sess = tf.Session()

    # transition function
    def run(self, w, worker):
        X, Y, n = worker.features, worker.labels, worker.n_samples
        assert(X.ndim == 2 and Y.ndim == 1)
        if not self.sess: self._initialize(self.dim)
        loss, df = self.sess.run([self.loss, self.grads],
                             feed_dict={self.x: X, self.y: Y, self.w: w})
        return loss/n, df/n
