#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2016-01-06 20:16:33
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-02-11 23:53:40
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.

from setuptools import Extension, setup
import os


sources = [
    'peregrine/src/array.cpp',
    'peregrine/src/lhac-py-gen.cpp',
    'peregrine/src/peregrine.cpp'
]

PYTHON_ROOT = '/Users/xtang/anaconda'
PYTHON_LINK = PYTHON_ROOT + '/lib'
PYTHON_INCLUDE = PYTHON_ROOT + '/include/python2.7'

link_flags = ['-Wall']
link_flags.append(
    '-Wl,-rpath,{PYTHON_ROOT} -L{PYTHON_LINK} -lpython2.7'
    .format(**locals())
)

flags = [
    '-fpic', '-fno-omit-frame-pointer',
    '-std=c++11', '-DUSE_CBLAS', '-m64', '-stdlib=libc++',
    '-I{PYTHON_INCLUDE}'.format(**locals())
]

os.environ.setdefault('CC', 'g++')
os.environ.setdefault('CXX', 'g++')
os.environ.setdefault('MACOSX_DEPLOYMENT_TARGET', '10.7')

peregrine_extension = Extension(
    'peregrine._peregrine',
    sources=sources,
    language="c++",
    include_dirs=[os.path.join('.', 'peregrine', 'include')],
    extra_compile_args=flags,
    extra_link_args=link_flags
)

setup(
    name = 'peregrine',
    ext_modules = [peregrine_extension]
)
