#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2015-12-17 23:01:45
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-07-05 11:47:57
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.


from pyspark import SparkConf, SparkContext
from sklearn.datasets import load_svmlight_file
from sklearn import datasets
import numpy as np


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





