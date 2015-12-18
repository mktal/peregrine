#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2015-12-17 15:01:49
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2015-12-17 15:03:34
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.


from models import LogReg, load_mnist
from cahow import train


data, target = load_mnist()
prob = LogReg(data, target, cached=True)
f, g = prob.eval_obj, prob.eval_grad
train(f, g, prob.shape[1], verbose=1)
