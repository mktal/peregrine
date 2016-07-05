#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2016-02-12 16:46:25
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-07-05 13:58:19
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.

import numpy as np
from utils import load_digits, load_mnist
from peregrine import descend


class LogregExecutor(object):
    """logistic regression using numpy"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.n_samples = self.features.shape[0]
        self.n_features = self.features.shape[1]

    @property
    def n_variables(self):
        """number of optimization variables"""
        return self.n_features

    def eval_obj(self, w):
        """evaluate objective value at w"""
        w = np.array(w, copy=False)
        X, Y = self.features, self.labels
        self.e_ywx = np.exp(Y*X.dot(w))
        return np.average(np.log1p(1/self.e_ywx))

    def eval_grad(self, w, df):
        """evaluate gradient at w and write it to df"""
        w = np.array(w, copy=False)
        df = np.array(df, copy=False)
        X, Y, n = self.features, self.labels, self.n_samples
        a = -Y/(1+self.e_ywx)/n
        (X*a[:,np.newaxis]).sum(axis=0, out=df)


class LogregTensorExecutor(object):
    """logistic regression using tensorflow"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.n_samples = self.features.shape[0]
        self.n_features = self.features.shape[1]
        self._initialize(self.n_features)

    @property
    def n_variables(self):
        """number of optimization variables"""
        return self.n_features

    def _initialize(self, dim):
        import tensorflow as tf
        self.w = tf.Variable(tf.zeros([dim]), name='model')
        self.x = tf.placeholder("float", [None, dim], name='features')
        self.y = tf.placeholder("float", [None], name='target')
        self.loss = -tf.reduce_mean(tf.log(tf.sigmoid(
                    self.y*tf.reduce_sum(self.x*self.w, 1))))
        self.grads = tf.gradients(self.loss, self.w)[0]
        self.sess = tf.Session()

    def eval_obj(self, w):
        """evaluate objective value at w"""
        w = np.array(w, copy=False)
        X, Y = self.features, self.labels
        f, self.df = self.sess.run([self.loss, self.grads],
                                   feed_dict={self.x: X, self.y: Y, self.w: w})
        return f

    def eval_grad(self, w, df):
        """evaluate gradient at w and write it to df"""
        df = np.array(df, copy=False)
        np.copyto(df, self.df)


X, y = load_mnist()
probs = []
probs.append(LogregExecutor(X, y))
try:
    import tensorflow as tf
    probs.append(LogregTensorExecutor(X, y))
except ImportError:
    print "No Tensoflow found: skip the example."


for prob in probs:
    descend(prob, initial_stepsize=0.0001, verbose=5,
            max_iter=10, l1_reg=0.002, precision='f')
# ------------------------------------------------------------------------------

