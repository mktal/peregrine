/*
* @Date:   2015-12-15 17:02:23
* @Last Modified by:   Xiaocheng Tang
* @Last Modified time: 2015-12-15 17:07:43
*
* Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
* All rights reserved.
*/



#include "array.h"

template <typename TypeValue>
void _init_array(py::module &m, const char* name) {
    py::class_<Array<TypeValue>> ary(m, "Array");

    ary.def(py::init<size_t>())
        /// Construct from a buffer
        .def("__init__", [](Array<TypeValue> &v, py::buffer b, bool copy = false) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<TypeValue>::value() || info.ndim != 1)
                throw std::runtime_error("Incompatible buffer format!");
            if (!copy) {
                std::cout << "construct without copy from python object" << std::endl;
                new (&v) Array<TypeValue>((TypeValue* )info.ptr, info.shape[0]);
            }
            else {
                std::cout << "construct with copy from python object" << std::endl;
                new (&v) Array<TypeValue>(info.shape[0]);
                memcpy(v.data(), info.ptr, sizeof(TypeValue) * v.size());
            }
        }, py::arg("b"), py::arg("copy") = false)

       .def("size", &Array<TypeValue>::size)

        /// Bare bones interface
       .def("__getitem__", [](const Array<TypeValue> &m, size_t i) {
            if (i >= m.size())
                throw py::index_error();
            return m(i);
        })
       .def("__setitem__", [](Array<TypeValue> &m, size_t i, TypeValue v) {
            if (i >= m.size())
                throw py::index_error();
            m(i) = v;
        })
       /// Provide buffer access
       .def_buffer([](Array<TypeValue> &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                              /* Pointer to buffer */
                sizeof(TypeValue),                         /* Size of one scalar */
                py::format_descriptor<TypeValue>::value(), /* Python struct-style format descriptor */
                1,                                     /* Number of dimensions */
                { m.size(), },                /* Buffer dimensions */
                { sizeof(TypeValue), }           /* Strides (in bytes) for each index */
            );
        });
}

void init_array(py::module &m) {
    _init_array<double>(m, "Array");
}
