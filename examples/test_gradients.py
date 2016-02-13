#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2015-12-15 15:23:56
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-02-12 17:30:41
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.


import unittest
from pyspark.mllib.util import MLUtils
from pyspark import SparkConf, SparkContext
import numpy as np
import pprint as pp
from utils import verify_gradient
from utils import load_digits
from utils import SparkController

from peregrine.objectives import logistic_regression
from peregrine.objectives import Worker


class LogRegTestCase(unittest.TestCase):

    @unittest.skip('.')
    def test_baseline(self):
        dataset = Worker(*load_digits())
        prob = logistic_regression(dataset, l2_reg=0.001)
        g1, g2 = verify_gradient(prob)
        d = np.linalg.norm(g1-g2)
        pp.pprint(d)
        self.assertAlmostEqual(d, 0.0, places=5)

    def test_tensorflow(self):
        dataset = Worker(*load_digits())
        # baseline = logistic_regression(dataset, l2_reg=0.001)
        tf = logistic_regression(dataset, l2_reg=0.001, tensorflow=True)
        g1, g2 = verify_gradient(tf)
        d = np.linalg.norm(g1-g2)
        pp.pprint(d)
        self.assertAlmostEqual(d, 0.0, places=5)



# class DistributedTrainTestCase(unittest.TestCase):

#     @classmethod
#     def setUp(cls):
#         conf = SparkConf().setMaster('local[*]').setAppName('SVRG_TEST')
#         cls.sc = SparkContext(conf=conf)

#     @classmethod
#     def tearDown(cls):
#         cls.sc.stop()

#     def _run_verify(self, prob):
#         g1, g2 = verify_gradient(prob, 1e-8)
#         d = np.linalg.norm(g1-g2)
#         pp.pprint(d)
#         self.assertAlmostEqual(d, 0.0, places=5)

#     @unittest.skip('.')
#     def test_LogRegDV(self):
#         sc = self.sc
#         dataset = MLUtils.loadLibSVMFile(sc, './data/a9a').cache()
#         prob = LogRegDV(dataset, cached=False, l2_reg=1)
#         g1, g2 = verify_gradient(prob)
#         d = np.linalg.norm(g1-g2)
#         pp.pprint(d)
#         self.assertAlmostEqual(d, 0.0, places=6)

#     # @unittest.skip('.')
#     def test_LogRegDM(self):
#         sc = self.sc
#         dataset = MLUtils.loadLibSVMFile(sc, './data/a9a')
#         prob = LogRegDM(dataset, cached=False, dense=False, l2_reg=0.001)
#         self._run_verify(prob)
#         prob = LogRegDM(dataset, cached=False, dense=True, l2_reg=0.001)
#         self._run_verify(prob)


if __name__ == '__main__':
    unittest.main()
