#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2016-01-06 20:16:33
# @Last Modified by:   Xiaocheng Tang
# @Last Modified time: 2016-04-21 14:48:20
#
# Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
# All rights reserved.

from setuptools import Extension, setup
from sys import platform as _platform
import os



sources = [
    'peregrine/src/array.cpp',
    'peregrine/src/lhac-py-gen.cpp',
    'peregrine/src/peregrine.cpp'
]

link_flags = ['-Wall', '-llapack', '-lblas', '-lpython2.7']

flags = [
    '-fpic', '-fno-omit-frame-pointer',
    '-std=c++11', '-DUSE_CBLAS', '-m64'
]

# MAC OS X
if _platform == "darwin":
    flags.append('-stdlib=libc++')
# elif _platform in ('linux', 'linux2'):
    # pass

os.environ.setdefault('CC', 'g++')
os.environ.setdefault('CXX', 'g++')
os.environ.setdefault('MACOSX_DEPLOYMENT_TARGET', '10.8')

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
