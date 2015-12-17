/*
    example/example.cpp -- pybind example plugin

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

void init_array(py::module &);
void init_matrix(py::module &);
void init_train(py::module &);

PYBIND11_PLUGIN(example) {
    py::module m("example", "pybind example plugin");

    init_array(m);
    init_matrix(m);
    init_train(m);

    return m.ptr();
}
