/*
* @Date:   2015-12-15 16:13:07
* @Last Modified by:   Xiaocheng Tang
* @Last Modified time: 2016-04-24 18:32:54
*
* Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
* All rights reserved.
*/


#include "array.h"
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <Objective.h>
#include <lhac.h>

template <typename T1>
class PyObjective : public Objective<PyObjective<T1>, T1>
{
public:
    inline size_t getDims() const { return _dim; }

    inline T1 computeObject(T1 *wnew) {
        py::object res = _f_func.call(Array<T1>(wnew, _dim));
        return res.cast<T1>();
    }

    inline void computeGradient(T1* wnew, T1* df) {
        _g_func.call(Array<T1>(wnew, _dim), Array<T1>(df, _dim));
    }

    PyObjective(py::object f_func, py::object g_func, size_t dim)
    : _f_func(f_func), _g_func(g_func), _dim(dim) {}

private:
    py::object _f_func;
    py::object _g_func;
    size_t _dim;
};

template <typename T1>
void _run(py::object f_func,
          py::object g_func,
          py::object final_func,
          size_t dim, Parameter* param)
{
    PyObjective<T1>* prob = new PyObjective<T1>(f_func, g_func, dim);

    LHAC<PyObjective<T1>, T1>* Alg = new LHAC<PyObjective<T1>, T1>(prob, param);
    Solution<T1>* sols = Alg->solve();
    final_func.call(Array<T1>(sols->w, dim));
    delete Alg;
    delete prob;

    return;
}

void _train(py::object f_func, py::object g_func, py::object final_func,
            size_t dim, const std::vector<float> &model_initial,
            float stepsize_initial = 1,
            size_t verbose = 2, double opt_tol = 1e-8, size_t max_iter = 500,
            size_t memory = 10, double l1_reg = 1e-6, char precision = 'f') {
    // shrink -> gama = gama / shrink
    double shrink = 4;
    unsigned long cd_rate = 6;
    // for greedy active set
    unsigned long work_size = 500;
    // active set strategy -- standard (default)
    unsigned long active_set = STD;
//    unsigned long active_set = GREEDY_ADDZERO;
    int method_flag = 4;
    double rho = 0.5;

    Parameter* param = new Parameter;
    param->l = memory;
    param->work_size = work_size;
    param->max_iter = max_iter;
    param->lmd = l1_reg;
    param->opt_outer_tol = opt_tol;
    param->verbose = verbose;
    param->shrink = shrink;
    param->rho = rho;
    param->cd_rate = cd_rate;
    param->active_set = active_set;
    param->method_flag = method_flag;
    param->stepsize_initial = stepsize_initial;
    param->w_initial = (model_initial.size()==dim) ? &model_initial[0] : NULL;

    if (precision == 'f') {
        std::cout << "Solving in single precision...." << std::endl;
        _run<float>(f_func, g_func, final_func, dim, param);
    }
    else {
        std::cout << "Solving in double precision...." << std::endl;
        _run<double>(f_func, g_func, final_func, dim, param);
    }
    delete param;
}


void init_train(py::module &m) {
    m.def("_train", &_train,
          py::arg("f_func"),
          py::arg("g_func"),
          py::arg("final_func"),
          py::arg("dim"),
          py::arg("model_initial"),
          py::arg("stepsize_initial") = 1,
          py::arg("verbose") = 2, py::arg("opt_tol") = 1e-8,
          py::arg("max_iter") = 500, py::arg("memory") = 10,
          py::arg("l1_reg") = 1e-6, py::arg("precision") = 'f');
}






