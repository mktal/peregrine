#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2016-02-12 17:01:51
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-04-21 23:27:17
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.


from utils import SparkController

from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils
from pyspark.ml.classification import LogisticRegression

from peregrine import descend
from peregrine.objectives import Worker
from peregrine.objectives import Executor

from logreg import logistic_regression
from logreg import logreg_local
from logreg import collect_one


with SparkController() as sc:
    data_path, npar = './data/a9a', 5
    dataset = MLUtils.loadLibSVMFile(sc, data_path, minPartitions=npar).cache()

    local_data = Worker.from_rows(dataset.collect(), dense=False)
    n, d = local_data.n_samples, local_data.n_features
    print '#samples: {n}; #features: {d}'.format(n=n, d=d)

    print 'Baseline: training in single node mode...'
    prob = Executor(local_data, n, d, collect_one,
                    logreg_local, cached=True, l2_reg=0.01)
    descend(prob, verbose=1, max_iter=30, l1_reg=0.005, precision='f')

    print 'Spark ({} partitions): training using peregrine...'.format(npar)
    prob = logistic_regression(dataset, dense=False, l2_reg=0.01)
    descend(prob, verbose=1, max_iter=30, l1_reg=0.005, precision='f')

    print 'Spark ({} partitions): training using mllib...'.format(npar)
    sqlContext = SQLContext(sc)
    lr = LogisticRegression(maxIter=300, regParam=0.02,
                            elasticNetParam=0.5, fitIntercept=False)
    lr.fit(dataset.toDF().replace(-1, 0, 'label').cache())

    print 'Spark/Tensorflow ({} partitions): training using peregrine...'.format(npar)
    prob = logistic_regression(dataset, l2_reg=0.01, tensorflow=True)
    descend(prob, verbose=1, max_iter=30, l1_reg=0.005, precision='f')

