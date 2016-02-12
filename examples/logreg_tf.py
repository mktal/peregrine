#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2016-02-11 22:22:00
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-02-11 23:01:08
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.

from pyspark.mllib.util import MLUtils
from utils import SparkController
from peregrine import descend
from peregrine import DistributedWithSpark
import tensorflow as tf


class LogRegWithTF(object):
    """logistic regression using tensorflow"""
    def __init__(self, dim=None):
        self._initialize(dim)

    def _initialize(self, dim=None):
        self.dim = dim
        if not dim: return
        self.w = tf.Variable(tf.zeros([dim]), name='model')
        self.x = tf.placeholder("float", [None, dim], name='features')
        self.y = tf.placeholder("float", [None], name='target')
        self.loss = -tf.reduce_sum(tf.log(tf.sigmoid(
                    self.y*tf.reduce_sum(self.x*self.w, 1))))
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())
        self.grads = tf.gradients(self.loss, self.w)[0]

    def run(self, w, worker):
        X, Y, n = worker.features, worker.labels, worker.n_samples
        assert(X.ndim == 2 and Y.ndim == 1)
        if self.dim is None: self._initialize(X.shape[1])
        return self.sess.run([self.loss, self.grads],
                             feed_dict={self.x: X, self.y: Y, self.w: w})


def peregrineLogReg(sc, data_path):
    dataset = MLUtils.loadLibSVMFile(sc, data_path, minPartitions=5).cache()
    prob = DistributedWithSpark(dataset, LogRegWithTF().run,
                                dense=True, cached=True, l2_reg=0.001)
    descend(prob, verbose=1, max_iter=30, l1_reg=0.001, precision='f')


if __name__ == '__main__':
    with SparkController() as sc:
        peregrineLogReg(sc, '/Users/xtang/Documents/peregrine/examples/data/a9a')

