#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2015-12-17 15:01:49
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-01-05 21:15:39
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.


from models import LogReg
from utils import load_mnist
from ..peregrine import descend


data, target = load_mnist()
prob = LogReg(data, target, cached=True)
descend(prob, prob.shape[1], verbose=1, precision='d')
