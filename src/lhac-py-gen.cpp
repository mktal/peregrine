/*
* @Date:   2015-12-15 16:13:07
* @Last Modified by:   Xiaocheng Tang
* @Last Modified time: 2015-12-16 20:04:29
*
* Copyright (c) 2016 Xiaocheng Tang <xiaocheng.t@gmail.com>
* All rights reserved.
*/


#include "array.h"
#include <pybind11/functional.h>
#include <Objective.h>
#include <lhac.h>

class PyObjective : public Objective<PyObjective>
{
public:
    size_t getDims() const {
        return _dim;
    }

    double computeObject(double *wnew) {
        // Array<double> a = Array<double>(wnew, _dim);
        Array<double> a(wnew, _dim);
        py::object res = _f_func.call(a);
        return res.cast<double>();
    }

    void computeGradient(double* wnew, double* df) {
        Array<double> a(wnew, _dim);
        Array<double> b(df, _dim);
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
    PyObjective* prob = new PyObjective(f_func, g_func, dim);

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

void train(py::object f_func, py::object g_func, size_t dim) {
    // verbose
    int verbose = 2;
    // precision
    double opt_outer_tol = 1e-8;
    // shrink -> gama = gama / shrink
    double shrink = 4;
    // max iterations
    int max_iter = 500;
    // sufficient decrease (default) or backtrack
    int sd_flag = 1;
    // max_cdpass = 1 + iter / cdrate
    unsigned long cd_rate = 6;
    // for greedy active set
    unsigned long work_size = 500;
    // active set strategy -- standard (default)
    unsigned long active_set = STD;
//    unsigned long active_set = GREEDY_ADDZERO;
    int loss = LOG;
    bool isCached = true;
    double lambda = 0.000001;
    double posweight = 1.0;
    // LBFGS limited memory parameter
    int limited_memory = 10;
    int method_flag = 4;
    double rho = 0.5;

    Parameter* param = new Parameter;
    param->l = limited_memory;
    param->work_size = work_size;
    param->max_iter = max_iter;
    param->lmd = lambda;
    param->max_inner_iter = 100;
    param->opt_inner_tol = 5*1e-6;
    param->opt_outer_tol = opt_outer_tol;
    param->max_linesearch_iter = 1000;
    param->bbeta = 0.5;
    param->ssigma = 0.001;
    param->verbose = verbose;
    param->sd_flag = sd_flag;
    param->shrink = shrink;
    param->fileName = "none";
    param->rho = rho;
    param->cd_rate = cd_rate;
    param->active_set = active_set;
    param->loss = loss;
    param->isCached = isCached;
    param->dense = 1;
    param->posweight = posweight;
    param->method_flag = method_flag;

    PyObjective* prob = new PyObjective(f_func, g_func, dim);

    LHAC<PyObjective>* Alg = new LHAC<PyObjective>(prob, param);
    Solution* sols = Alg->solve();

}


void init_train(py::module &m) {
    m.def("train", &train);
    m.def("test_objective", &test_objective);
}






