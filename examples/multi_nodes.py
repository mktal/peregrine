#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2016-02-12 17:01:51
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-02-12 23:26:11
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.

from pyspark.mllib.util import MLUtils
from utils import SparkController
from peregrine import descend
from peregrine.objectives import logistic_regression


with SparkController() as sc:
    data_path = './data/a9a'
    dataset = MLUtils.loadLibSVMFile(sc, data_path, minPartitions=5).cache()
    prob = logistic_regression(dataset, dense=False, l2_reg=0.001)
    descend(prob, verbose=0, max_iter=30, l1_reg=0.001, precision='f')

    prob = logistic_regression(dataset, l2_reg=0.001, tensorflow=True)
    descend(prob, verbose=0, max_iter=50, l1_reg=0.001, precision='f')
