#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2016-01-04 20:21:11
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-01-05 21:15:24
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.


__all__ = ['descend']

from ._peregrine import _train


def descend(objective, verbose=2, opt_tol=1e-8, max_iter=500,
            memory=10, l1_reg=1e-6, precision='f', **kwargs):
    """Proximal quasi-Newton (L-BFGS) minimizer for composite function

    Parameters
    ----------

    objective : the objective function to be minimized. Required to implement
        two functions eval_obj (function value) and eval_obj (gradient value)

    dim : int, dimension of the objective function

    verbose : int, optional (default=2)
        set verbosity of the output. Set to 0 for silence

    opt_tol : float, optional (default=1e-8)
        set tolerance of termination criterion
        ``final ista step size <= eps*(initial ista step size)``

    max_iter : int, optional (default=500)
        max number of iterations

    memory : int, optional (default=10)
        limited memory of L-BFGS. Use cache size to the order of
        ``2*memory*dim*sizeof(precision)``
        to approximate the curvature of objective for faster convergence

    l1_reg : float, optional (default=1e-6)

    """
    f, g, dim= objective.eval_obj, objective.eval_grad, objective.shape[1]
    return _train(f, g, dim, verbose, opt_tol,
                  max_iter, memory, l1_reg, precision)
