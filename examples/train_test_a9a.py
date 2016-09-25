#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2016-09-23 14:57:46
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-09-25 14:34:18
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.


import numpy as np
from scipy.sparse import hstack
from peregrine import descend

from sklearn.datasets import load_svmlight_file
X, y = load_svmlight_file("data/a9a")

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.33, random_state=66)

class LogregExecutor(object):
    """logistic regression using numpy"""
    def __init__(self, features, labels, fit_intercept = True):
        self.fit_intercept = fit_intercept
        self.features = features
        self.labels = labels
        self.n_samples = self.features.shape[0]
        self.n_features = self.features.shape[1]
        if fit_intercept:
            self.n_features += 1

    @property
    def n_variables(self):
        """number of optimization variables"""
        return self.n_features

    def eval_obj(self, w):
        """evaluate objective value at w"""
        w = np.array(w, copy=False)
        X, Y = self.features, self.labels
        wa, wb = (w[:-1], w[-1]) if self.fit_intercept else (w, 0)
        self.e_ywx = np.exp(Y*(X.dot(wa) + wb))
        return np.average(np.log1p(1/self.e_ywx))

    def eval_grad(self, w, df):
        """evaluate gradient at w and write it to df"""
        w = np.array(w, copy=False)
        df = np.array(df, copy=False)
        X, Y, n = self.features, self.labels, self.n_samples
        a = -Y/(1+self.e_ywx)/n
        db = [np.sum(a)] if self.fit_intercept else []
        np.copyto(df, np.hstack([X.transpose().dot(a)] + db))

    def final(self, model):
        self._model = np.array(model, copy=True)

    def predict(self, X, y=None, threshold=0):
        _x = X
        if self.fit_intercept:
            _x = hstack((_x, np.ones((_x.shape[0], 1))))
        y_hat = _x.dot(self._model)
        if threshold is not None:
            assert threshold >= -1
            y_hat[y_hat<threshold] = -1
            y_hat[y_hat>=threshold] = 1
        return y_hat

prob = LogregExecutor(X_train, y_train, fit_intercept = False)
descend(prob, initial_stepsize=0.0001, verbose=1,
        max_iter=20, l1_reg=0.00002, precision='f')
print "Accuracy: {}".format(np.mean(prob.predict(X_test) == y_test))



