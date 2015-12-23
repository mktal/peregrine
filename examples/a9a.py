#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2015-12-17 21:51:20
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2015-12-22 20:24:50
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
from models import LogRegD
from models import LogRegDD
from cahow import train


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
    dataset = MLUtils.loadLibSVMFile(sc, data_path, minPartitions=2).cache()
    prob = LogRegDD(dataset, cached=True)
    f, g = prob.eval_obj, prob.eval_grad
    train(f, g, prob.shape[1], verbose=1, max_iter=30, l1_reg=0.001)


if __name__ == '__main__':
    with SparkController() as sc:
        # sparkLogReg(sc, './data/a9a')
        cahowLogReg(sc, './data/a9a')











