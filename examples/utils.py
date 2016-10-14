#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2015-12-17 23:01:45
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-10-14 11:02:22
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.


from pyspark import SparkConf, SparkContext
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from sklearn import datasets
import numpy as np
import os


import errno
def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


class SparkController(object):
  """docstring for SparkController"""
  def __init__(self, sc=None, master='local[*]', name='test'):
    self.master = master
    self.name = name
    self.sc = sc

  def __enter__(self):
    if self.sc is None:
      conf = SparkConf().setMaster(self.master)
      conf = conf.setAppName(self.name)
      self.sc = SparkContext(conf=conf)
    return self.sc

  def __exit__(self, type, value, tb):
    if self.sc is not None:
      self.sc.stop()


def verify_gradient(model, eps=0.0001):
    f, g = model.eval_obj, model.eval_grad
    d = model.n_variables
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
  data, target = load_svmlight_file('./data/mnist_0_6')
  print('load mnist for classification: {}'.format(data.shape))
  return np.asarray(data.todense()), target


def subsample_to_file(svm_file, out_dir, out_name, multilabel=False,
                      row_ratio=0.5, col_ratio=0.3, random_state=12):
  """
  Example:

  '''python
     # run the following command in the current directory will create a
     # `tmp` folder, if not already exists, and generate a file called
     # `a9a_sub` from the original file `./data/a9a`. Both files are
     # in libsvm format.
     subsample_to_file("./data/a9a", "./tmp", "a9a_sub")
     # read the subsampled file and make sure its number of rows is half of
     # that of a9a and its number of column is roughly third of a9a (123)
     X, y = load_svmlight_file('./tmp/a9a_sub')
     assert X.shape == (16280, 36)
  '''

  """
  assert 1 >= row_ratio > 0, \
         "Row ratio {row_ratio} must be (0, 1]" \
         .format(**locals())
  assert 1 >= col_ratio > 0, \
         "Col ratio {col_ratio} must be (0, 1]" \
         .format(**locals())
  X, y = load_svmlight_file(svm_file, multilabel=multilabel)
  n, m = X.shape
  subn = int(n*row_ratio)
  subm = int(m*col_ratio)
  rst = np.random.RandomState(random_state)
  ridx = rst.choice(n, subn, replace=False)
  cidx = rst.choice(m, subm, replace=False)
  mkdir_p(out_dir)
  out_file = os.path.join(out_dir, out_name)
  dump_svmlight_file(X[ridx,:][:,cidx], y[ridx],
                     out_file, multilabel=multilabel)

