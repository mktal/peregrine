#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2015-12-15 15:23:56
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2015-12-16 00:35:20
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.


import unittest
from example import Matrix, Array, train
import numpy as np
import pprint as pp
from sklearn import datasets
from sklearn.datasets import load_svmlight_file


def verify_gradient(model, eps=0.0001):
    f, g = model.eval_obj, model.eval_grad
    d = model.shape[1]
    w = np.random.random((d,))

    f0 = f(w)
    g0_ = auto_grad(f, w, eps, f0)
    g0 = np.zeros_like(w)
    g(w, g0)
    return g0, g0_


def auto_grad(f_eval, w, eps=0.0001, f0=None, **kwargs):
    W = eps*np.eye(w.shape[0]) + w
    f0 = f_eval(w, **kwargs) if f0 is None else f0
    return np.array([(f_eval(w_, **kwargs) - f0) / eps for w_ in W])


def load_digits():
    train = datasets.load_digits()
    data, target = train.data, train.target.copy()
    target[target != 0] = -1
    target[target == 0] = 1
    print('load digits for classification: {}'.format(data.shape))
    return data, target


def load_mnist():
  data, target = load_svmlight_file('/Users/xtang/data/mnist_0_6')
  print('load mnist for classification: {}'.format(data.shape))
  return np.asarray(data.todense()), target


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
        np.dot(a, self.X, out=df)
        return



@unittest.skip('.')
class MatrixTestCase(unittest.TestCase):

    def test_number_init(self):
        m = Matrix(5, 5)
        self.assertEqual(m[2, 3], 0)
        m[2, 3] = 4
        self.assertEqual(m[2, 3], 4)

    def test_numpy_wrap(self):
        m = Matrix(5, 5)
        m[2, 3] = 4
        a = np.array(m, copy=False)
        self.assertEqual(m[2, 3], a[2, 3])
        m[2, 3] = 5
        self.assertEqual(m[2, 3], a[2, 3])

    def test_numpy_init(self):
        a = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float64)
        # not copy by default
        a_alias = Matrix(a)
        a_copy = Matrix(a, copy=True)
        self.assertEqual(a_alias[1, 2], a_copy[1, 2])
        a[1, 2] = 11
        self.assertNotEqual(a_alias[1, 2], a_copy[1, 2])
        self.assertEqual(a_alias[1, 2], a[1, 2])

@unittest.skip('.')
class ArrayTestCase(unittest.TestCase):

    def test_number_init(self):
        a = Array(3)
        self.assertEqual(a[1], 0)
        a[1] = 3
        self.assertEqual(a[1], 3)

    def test_numpy_wrap(self):
        a = Array(3)
        a[1] = 3
        a_ = np.array(a, copy=False)
        self.assertEqual(a[1], a_[1])
        a_[1] = 5
        self.assertEqual(a[1], a_[1])

    def test_numpy_init(self):
        a = np.array([2, 3, 4]).astype(np.float64)
        a_alias = Array(a)
        a_copy = Array(a, copy=True)
        self.assertEqual(a_alias[1], a_copy[1])
        a[1] = 11
        self.assertEqual(a_alias[1], a[1])
        self.assertNotEqual(a_alias[1], a_copy[1])


class TrainTestCase(unittest.TestCase):

    @unittest.skip('.')
    def test_Objective(self):
        def f(w):
            a = np.array(w, copy=False)
            pp.pprint(a)
            a[0] += 1
            return a[0]

        def g(w, out):
            a = np.array(w, copy=False)
            b = np.array(out, copy=False)
            pp.pprint(b)
            np.multiply(a, 2, out=b)
        from example import test_objective
        test_objective(f, g, 3)

    # @unittest.skip('.')
    def test_LogReg(self):
        data, target = load_digits()
        prob = LogReg(data, target)
        g1, g2 = verify_gradient(prob)
        d = np.linalg.norm(g1-g2)
        pp.pprint(d)
        self.assertAlmostEqual(d, 0.0, places=7)

    # @unittest.skip('.')
    def test_Train(self):
        # data, target = load_digits()
        data, target = load_mnist()
        prob = LogReg(data, target, cached=True)
        f, g = prob.eval_obj, prob.eval_grad
        train(f, g, prob.shape[1])






if __name__ == '__main__':
    unittest.main()
