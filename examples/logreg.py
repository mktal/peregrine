#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2016-01-05 21:51:20
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-02-11 00:05:58
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.

from pyspark import SparkConf, SparkContext
from pyspark.mllib.util import MLUtils
from utils import SparkController
from ..peregrine import descend
from ..peregrine import DistributedWithSpark
import numpy as np


def logreg_trans(w, row):
    """ Log Reg Single row (dense or sparse). """
    X, Y, n = row.features, row.labels, row.n_samples
    e_ywx = np.exp(Y*X.dot(w))
    loss = np.average(np.log1p(1/e_ywx))
    a = -Y/(1+e_ywx)/n
    # df = np.dot(a, X)
    df = X.transpose().dot(a)
    return loss, df


def peregrineLogReg(sc, data_path):
    dataset = MLUtils.loadLibSVMFile(sc, data_path, minPartitions=8).cache()
    prob = DistributedWithSpark(dataset, logreg_trans,
                                cached=True, l2_reg=0.001)
    descend(prob, verbose=1, max_iter=30, l1_reg=0.001, precision='f')


if __name__ == '__main__':
    with SparkController() as sc:
        peregrineLogReg(sc, '/Users/xtang/Documents/cahow/examples/data/a9a')

