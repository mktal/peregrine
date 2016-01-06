/*
    example/example.cpp -- pybind example plugin

    Copyright (c) 2015 Wenzel Jakob <wenzel@inf.ethz.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <peregrine.h>

void init_array(py::module &);
void init_train(py::module &);

PYBIND11_PLUGIN(_peregrine) {
    py::module m("_peregrine", "Proximal L-BFGS solver for composite functions.");

    init_array(m);
    init_train(m);

    return m.ptr();
}
