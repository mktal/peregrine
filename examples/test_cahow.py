#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2015-12-15 15:23:56
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2015-12-25 00:49:33
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.


import unittest
from cahow import Matrix, Array, train
from pyspark.mllib.util import MLUtils
from pyspark import SparkConf, SparkContext
import numpy as np
import pprint as pp
from models import LogReg
from models import LogRegDV
from models import LogRegDM
from utils import verify_gradient
from utils import load_digits
from utils import SparkController


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

    @unittest.skip('.')
    def test_LogReg(self):
        data, target = load_digits()
        prob = LogReg(data, target)
        g1, g2 = verify_gradient(prob)
        d = np.linalg.norm(g1-g2)
        pp.pprint(d)
        self.assertAlmostEqual(d, 0.0, places=7)


class DistributedTrainTestCase(unittest.TestCase):

    @classmethod
    def setUp(cls):
        conf = SparkConf().setMaster('local[*]').setAppName('SVRG_TEST')
        cls.sc = SparkContext(conf=conf)

    @classmethod
    def tearDown(cls):
        cls.sc.stop()

    @unittest.skip('.')
    def test_LogRegDV(self):
        sc = self.sc
        dataset = MLUtils.loadLibSVMFile(sc, './data/a9a').cache()
        prob = LogRegDV(dataset, cached=False)
        g1, g2 = verify_gradient(prob)
        d = np.linalg.norm(g1-g2)
        pp.pprint(d)
        self.assertAlmostEqual(d, 0.0, places=6)

    # @unittest.skip('.')
    def test_LogRegDM(self):
        sc = self.sc
        dataset = MLUtils.loadLibSVMFile(sc, './data/a9a')
        prob = LogRegDM(dataset, cached=False, dense=False)
        g1, g2 = verify_gradient(prob)
        d = np.linalg.norm(g1-g2)
        pp.pprint(d)
        self.assertAlmostEqual(d, 0.0, places=6)


if __name__ == '__main__':
    unittest.main()
