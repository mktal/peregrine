/*
* @Author: Xiaocheng Tang
* @Date:   2015-12-15 15:25:08
* @Last Modified by:   Xiaocheng Tang
* @Last Modified time: 2015-12-17 15:11:12
*/

#include "cahow.h"

template <typename TypeValue>
class Matrix {
public:
    Matrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {
        std::cout << "Value constructor: Creating a " << rows << "x" << cols << " matrix " << std::endl;
        m_data = new TypeValue[rows*cols];
        memset(m_data, 0, sizeof(TypeValue) * rows * cols);
    }

    Matrix(TypeValue* data, size_t rows, size_t cols) : m_data(data), m_rows(rows), m_cols(cols) {
        std::cout << "Value constructor no copy: Creating a " << rows << "x" << cols << " matrix " << std::endl;
        owner = false;
    }

    Matrix(const Matrix &s) : m_rows(s.m_rows), m_cols(s.m_cols) {
        std::cout << "Copy constructor: Creating a " << m_rows << "x" << m_cols << " matrix " << std::endl;
        m_data = new TypeValue[m_rows * m_cols];
        memcpy(m_data, s.m_data, sizeof(TypeValue) * m_rows * m_cols);
    }

    Matrix(Matrix &&s) : m_rows(s.m_rows), m_cols(s.m_cols), m_data(s.m_data) {
        std::cout << "Move constructor: Creating a " << m_rows << "x" << m_cols << " matrix " << std::endl;
        s.m_rows = 0;
        s.m_cols = 0;
        s.m_data = nullptr;
    }

    ~Matrix() {
        if (owner) {
            std::cout << "Freeing a " << m_rows << "x" << m_cols << " matrix " << std::endl;
            delete[] m_data;
        }
        else
            std::cout << "not freeing" << std::endl;
    }

    Matrix &operator=(const Matrix &s) {
        std::cout << "Assignment operator : Creating a " << s.m_rows << "x" << s.m_cols << " matrix " << std::endl;
        delete[] m_data;
        m_rows = s.m_rows;
        m_cols = s.m_cols;
        m_data = new TypeValue[m_rows * m_cols];
        memcpy(m_data, s.m_data, sizeof(TypeValue) * m_rows * m_cols);
        return *this;
    }

    Matrix &operator=(Matrix &&s) {
        std::cout << "Move assignment operator : Creating a " << s.m_rows << "x" << s.m_cols << " matrix " << std::endl;
        if (&s != this) {
            delete[] m_data;
            m_rows = s.m_rows; m_cols = s.m_cols; m_data = s.m_data;
            s.m_rows = 0; s.m_cols = 0; s.m_data = nullptr;
        }
        return *this;
    }

    TypeValue operator()(size_t i, size_t j) const {
        return m_data[i*m_cols + j];
    }

    TypeValue &operator()(size_t i, size_t j) {
        return m_data[i*m_cols + j];
    }

    TypeValue *data() { return m_data; }

    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
private:
    size_t m_rows;
    size_t m_cols;
    bool owner = true;
    TypeValue *m_data;
};


template <typename TypeValue>
void _init_matrix(py::module &m) {
    py::class_<Matrix<TypeValue>> mtx(m, "Matrix");

    mtx.def(py::init<size_t, size_t>())
        /// Construct from a buffer
        .def("__init__", [](Matrix<TypeValue> &v, py::buffer b, bool copy = false) {
            py::buffer_info info = b.request();
            if (info.format != py::format_descriptor<TypeValue>::value() || info.ndim != 2)
                throw std::runtime_error("Incompatible buffer format!");
            if (!copy) {
                std::cout << "construct without copy from python object" << std::endl;
                new (&v) Matrix<TypeValue>((TypeValue* )info.ptr, info.shape[0], info.shape[1]);
            }
            else {
                std::cout << "construct with copy from python object" << std::endl;
                new (&v) Matrix<TypeValue>(info.shape[0], info.shape[1]);
                memcpy(v.data(), info.ptr, sizeof(TypeValue) * v.rows() * v.cols());
            }
        }, py::arg("b"), py::arg("copy") = false)

       .def("rows", &Matrix<TypeValue>::rows)
       .def("cols", &Matrix<TypeValue>::cols)

        /// Bare bones interface
       .def("__getitem__", [](const Matrix<TypeValue> &m, std::pair<size_t, size_t> i) {
            if (i.first >= m.rows() || i.second >= m.cols())
                throw py::index_error();
            return m(i.first, i.second);
        })
       .def("__setitem__", [](Matrix<TypeValue> &m, std::pair<size_t, size_t> i, TypeValue v) {
            if (i.first >= m.rows() || i.second >= m.cols())
                throw py::index_error();
            m(i.first, i.second) = v;
        })
       /// Provide buffer access
       .def_buffer([](Matrix<TypeValue> &m) -> py::buffer_info {
            return py::buffer_info(
                m.data(),                              /* Pointer to buffer */
                sizeof(TypeValue),                         /* Size of one scalar */
                py::format_descriptor<TypeValue>::value(), /* Python struct-style format descriptor */
                2,                                     /* Number of dimensions */
                { m.rows(), m.cols() },                /* Buffer dimensions */
                { sizeof(TypeValue) * m.rows(),            /* Strides (in bytes) for each index */
                  sizeof(TypeValue) }
            );
        });
}


void init_matrix(py::module &m) {
    _init_matrix<double>(m);
}
