#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2015-12-17 21:51:20
# @Last Modified by:   xtang
# @Last Modified time: 2016-01-06 22:44:24
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.

from pyspark import SparkConf, SparkContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SQLContext
from pyspark.sql import Row
from utils import SparkController
from models import LogRegDV
from models import LogRegDM
from ..peregrine import descend


def transformLabel(row):
    label, features = row.label, row.features
    label = 0. if label <= 0 else 1.
    return Row(label=label, features=features)

def sparkLogReg(sc, data_path):
    sqlContext = SQLContext(sc)
    # df = sqlContext.read.format('libsvm').load(data_path)
    df = MLUtils.loadLibSVMFile(sc, data_path, minPartitions=8).toDF().cache()
    lr = LogisticRegression(maxIter=300, regParam=0.001, elasticNetParam=1.,
                            fitIntercept=False)
    lr.fit(df.replace(-1, 0, 'label').cache())

def cahowLogReg(sc, data_path):
    dataset = MLUtils.loadLibSVMFile(sc, data_path, minPartitions=8).cache()
    prob = LogRegDM(dataset, cached=True, l2_reg=0.)
    descend(prob, verbose=1, max_iter=30, l1_reg=0.001)


if __name__ == '__main__':
    # import sys
    # print sys.argv[1]
    with SparkController() as sc:
        sparkLogReg(sc, '/Users/xtang/Documents/cahow/examples/data/a9a')
        # cahowLogReg(sc, '/Users/xtang/Documents/cahow/examples/data/a9a')











