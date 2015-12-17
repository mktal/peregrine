/*
* @Date:   2015-12-15 16:13:07
* @Last Modified by:   Xiaocheng Tang
* @Last Modified time: 2015-12-16 23:50:00
*
* Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
* All rights reserved.
*/


#include "array.h"
#include <pybind11/functional.h>
#include <Objective.h>
#include <lhac.h>

template <typename T1>
class PyObjective : public Objective<PyObjective<T1>, T1>
{
public:
    size_t getDims() const {
        return _dim;
    }

    T1 computeObject(T1 *wnew) {
        // Array<T1> a = Array<T1>(wnew, _dim);
        Array<T1> a(wnew, _dim);
        py::object res = _f_func.call(a);
        return res.cast<T1>();
    }

    void computeGradient(T1* wnew, T1* df) {
        Array<T1> a(wnew, _dim);
        Array<T1> b(df, _dim);
        py::object res = _g_func.call(a, b);
        return;
    }

    PyObjective(py::object f_func, py::object g_func, size_t dim) : _f_func(f_func), _g_func(g_func), _dim(dim) {}

private:
    py::object _f_func;
    py::object _g_func;
    size_t _dim;
};


void test_objective(py::object f_func, py::object g_func, size_t dim) {
    double* weights = new double[dim];
    for (int i=0; i<dim; i++) {
        weights[i] = i;
    }
    double* grad = new double[dim];
    memset(grad, 0, sizeof(double) * dim);
    PyObjective<double>* prob = new PyObjective<double>(f_func, g_func, dim);

    // prob->computeGradient(weights, grad);
    // for (int i=0; i<dim; i++) {
    //     std::cout << i << ',' << grad[i] << std::endl;
    // }

    double fval;
    fval = prob->computeObject(weights);
    std::cout << fval << std::endl;
    fval = prob->computeObject(weights);
    std::cout << fval << std::endl;


}

void train(py::object f_func, py::object g_func, size_t dim,
           size_t verbose = 2, double opt_tol = 1e-8, size_t max_iter = 500,
           size_t memory = 10, double lambda = 1e-6) {
    // shrink -> gama = gama / shrink
    double shrink = 4;
    unsigned long cd_rate = 6;
    // for greedy active set
    unsigned long work_size = 500;
    // active set strategy -- standard (default)
    unsigned long active_set = STD;
//    unsigned long active_set = GREEDY_ADDZERO;
    // LBFGS limited memory parameter
    int method_flag = 4;
    double rho = 0.5;

    Parameter* param = new Parameter;
    param->l = memory;
    param->work_size = work_size;
    param->max_iter = max_iter;
    param->lmd = lambda;
    param->opt_outer_tol = opt_tol;
    param->verbose = verbose;
    param->shrink = shrink;
    param->rho = rho;
    param->cd_rate = cd_rate;
    param->active_set = active_set;
    param->method_flag = method_flag;

    PyObjective<float>* prob = new PyObjective<float>(f_func, g_func, dim);

    LHAC<PyObjective<float>, float>* Alg = new LHAC<PyObjective<float>, float>(prob, param);
    Solution<float>* sols = Alg->solve();
}


void init_train(py::module &m) {
    m.def("train", &train,
          py::arg("f_func"), py::arg("g_func"), py::arg("dim"),
          py::arg("verbose") = 2, py::arg("opt_tol") = 1e-8,
          py::arg("max_iter") = 500, py::arg("memory") = 10,
          py::arg("lambda") = 1e-6);
    m.def("test_objective", &test_objective);
}






