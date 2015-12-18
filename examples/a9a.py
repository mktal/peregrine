#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2015-12-17 21:51:20
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2015-12-17 23:48:21
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.

from pyspark import SparkConf, SparkContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from utils import SparkController
from models import LogRegD
from cahow import train


with SparkController() as sc:
    dataset = MLUtils.loadLibSVMFile(sc, './data/a9a', minPartitions=8)
    prob = LogRegD(dataset, cached=True)
    f, g = prob.eval_obj, prob.eval_grad
    train(f, g, prob.shape[1], verbose=1, max_iter=30, l1_reg=0.001)










