//
//  lhac.h
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 1/31/13.
//  Copyright (c) 2013 Xiaocheng Tang. All rights reserved.
//

#ifndef __LHAC_v1__lhac__
#define __LHAC_v1__lhac__

#include "Lbfgs.h"
#include "Objective.h"
#include <math.h>
#include "linalg.h"
#include "timing.h"
#include "Parameter.h"
#ifdef _OPENMP
#include <omp.h>
#endif


#define MAX_LENS 1024

#define __MATLAB_API__


enum { LHAC_MSG_NO=0, LHAC_MSG_NEWTON, LHAC_MSG_SD, LHAC_MSG_CD, LHAC_MSG_MAX };


enum{  GREEDY= 1, STD, GREEDY_CUTZERO, GREEDY_CUTGRAD, GREEDY_ADDZERO, STD_CUTGRAD, STD_CUTGRAD_AGGRESSIVE };



struct Func {
    double f;
    double g;
    double val; // f + g

    inline void add(const double _f, const double _g) {
        f = _f;
        g = _g;
        val = f + g;
    };
};

template <typename T1>
struct Solution {
    T1* t;
    T1* fval;
    T1* normgs;
    int* niter;
    unsigned long* numActive;

    T1* w;
    unsigned long p; //dimension of w

    T1 cdTime;
    T1 lsTime;
    T1 lbfgsTime1;
    T1 lbfgsTime2;
    unsigned long size; // max_newton_iter

    unsigned long ngval;
    unsigned long nfval;
    unsigned long nls; // # of line searches
    T1 gvalTime;
    T1 fvalTime;

    inline void addEntry(T1 objval, T1 normsg, T1 elapsedTime,
                         int iter, unsigned long _numActive) {
        fval[size] = objval;
        normgs[size] = normsg;
        t[size] = elapsedTime;
        niter[size] = iter;
        numActive[size] = _numActive;
        (size)++;
    };

    inline void finalReport(const int error, T1* wfinal) {
        memcpy(w, wfinal, p*sizeof(T1));
        unsigned long last = size - 1;
        printf(
               "=========================== final report ========================\n"
               );
        if (error)
            printf("Terminated!\n");
        else
            printf("Optimal!\n");
        printf(
               "Best objective value found %+.6e\n"
               "In %3d iterations (%.4e seconds)\n"
               "With a precision of: %+.4e\n"
               "=================================================================\n",
               fval[last], niter[last], t[last], normgs[last] / normgs[0]
               );
    };

    Solution(unsigned long max_iter, unsigned long _p) {
        fval = new T1[max_iter];
        normgs = new T1[max_iter];
        t = new T1[max_iter];
        niter = new int[max_iter];
        numActive = new unsigned long[max_iter];
        cdTime = 0;
        lbfgsTime1 = 0;
        lbfgsTime2 = 0;
        lsTime = 0;
        ngval = 0;
        nfval = 0;
        gvalTime = 0.0;
        fvalTime = 0.0;
        nls = 0;
        size = 0;
        p = _p;
        w = new T1[p];
    };

    ~Solution() {
        delete [] w;
        delete [] fval;
        delete [] normgs;
        delete [] t;
        delete [] niter;

        return;
    };
};



template <typename InnerSolver, typename T1>
class Subproblem
{
public:
    inline void build(LBFGS<T1>* lR, T1* grad,
                      work_set_struct* work_set) {
        return static_cast<InnerSolver*>(this)->build(lR, grad, work_set);
    };

    inline T1 objective_value(const T1 gama) {
        return static_cast<InnerSolver*>(this)->objective_value(gama);
    };

    inline const T1* solve(const T1* w_prev,
                               const unsigned short k,
                               T1 gama) {
        return static_cast<InnerSolver*>(this)->solve(w_prev, k, gama);
    };

    virtual ~Subproblem() {};

};

template <typename T1>
class CoordinateDescent: public Subproblem<CoordinateDescent<T1>, T1>
{
public:
    CoordinateDescent(const Parameter* const _param, unsigned long _p): p(_p) {
        lmd = _param->lmd;
        l = _param->l;
        cd_rate = _param->cd_rate;
        msgFlag = _param->verbose;
        D = new T1[p];
        H_diag = new T1[p]; // p
        d_bar = new T1[2*l]; // 2*l
    };

    ~CoordinateDescent() {
        delete [] D;
        delete [] H_diag;
        delete [] d_bar;
    };


    void build(LBFGS<T1>* lR, T1* grad,
               work_set_struct* work_set) {
        Q = lR->Q;
        Q_bar = lR->Q_bar;
        m = lR->m;
        L_grad = grad;
        permut = work_set->permut;
        idxs = work_set->idxs;
        numActive = work_set->numActive;
        gama0 = lR->gama;
        buffer = lR->buff;
        memset(D, 0, p*sizeof(T1));
        memset(d_bar, 0, 2*l*sizeof(T1));
        for (unsigned long k = 0, i = 0; i < work_set->numActive; i++, k += m) {
            H_diag[i] = gama0;
            for (unsigned long j = 0; j < m; j++)
                H_diag[i] -= Q_bar[k+j]*Q[k+j];
        }
    };

    T1 objective_value(const T1 gama) {
        T1 order1 = lcddot((int)p, D, 1, L_grad, 1);
        T1 order2 = 0;
        int cblas_M = (int) numActive;
        int cblas_N = (int) m;
        lcdgemv(CblasColMajor, CblasTrans, Q, d_bar, buffer, cblas_N, cblas_M, cblas_N);
        T1 vp = 0;
        for (unsigned long ii = 0; ii < numActive; ii++) {
            unsigned long idx = idxs[ii].j;
            unsigned long idx_Q = permut[ii];
            vp += D[idx]*buffer[idx_Q];
        }
        order2 = gama*lcddot((int)p, D, 1, D, 1)-vp;
        order2 = order2*0.5;
        return order1 + order2;
    }

    const T1* solve(const T1* w_prev,
                        const unsigned short k,
                        T1 gama) {
        T1 dH_diag = gama-gama0;
        unsigned long max_cd_pass = 1 + k / cd_rate;
        for (unsigned long cd_pass = 1; cd_pass <= max_cd_pass; cd_pass++) {
            T1 diffd = 0;
            T1 normd = 0;
            for (unsigned long ii = 0; ii < numActive; ii++) {
                unsigned long rii = ii;
                unsigned long idx = idxs[rii].j;
                unsigned long idx_Q = permut[rii];
                unsigned long Q_idx_m = idx_Q*m;
                T1 Qd_bar = lcddot(m, &Q[Q_idx_m], 1, d_bar, 1);
                T1 Hd_j = gama*D[idx] - Qd_bar;
                T1 Hii = H_diag[idx_Q] + dH_diag;
                T1 G = Hd_j + L_grad[idx];
                T1 Gp = G + lmd;
                T1 Gn = G - lmd;
                T1 wpd = w_prev[idx] + D[idx];
                T1 Hwd = Hii * wpd;
                T1 z = -wpd;
                if (Gp <= Hwd) z = -Gp/Hii;
                if (Gn >= Hwd) z = -Gn/Hii;
                D[idx] = D[idx] + z;
                for (unsigned long k = Q_idx_m, j = 0; j < m; j++)
                    d_bar[j] += z*Q_bar[k+j];
                diffd += fabs(z);
                normd += fabs(D[idx]);
            }
            if (msgFlag >= LHAC_MSG_CD) {
                printf("\t\t Coordinate descent pass %ld:   Change in d = %+.4e   norm(d) = %+.4e\n",
                       cd_pass, diffd, normd);
            }
        }
        return D;
    };

private:
    /* own */
    T1* D;
    T1* d_bar;
    T1* H_diag;

    T1* Q;
    T1* Q_bar;
    T1* L_grad;
    T1* buffer;
    unsigned long* permut;
    ushort_pair_t* idxs;
    T1 lmd;
    T1 gama0;
    unsigned long cd_rate;
    unsigned long l;
    unsigned long numActive;
    int msgFlag;
    unsigned long p;
    unsigned short m;

};


template <typename Derived, typename T1>
class LHAC
{
public:

    LHAC(Objective<Derived, T1>* _mdl, const Parameter* const _param)
    : mdl(_mdl), param(_param) {
        p = mdl->getDims();
        obj = new Func;

        l = param->l;
        opt_outer_tol = param->opt_outer_tol;
        max_iter = param->max_iter;
        lmd = param->lmd;
        msgFlag = param->verbose;

        w_prev = new T1[p];
        w = new T1[p];
        L_grad_prev = new T1[p];
        L_grad = new T1[p];
        D = new T1[p];
        H_diag = new T1[p]; // p
        d_bar = new T1[2*param->l]; // 2*l

        /* initiate */
        if (_param->w_initial == NULL) {
            memset(w, 0, p*sizeof(T1));
            memset(w_prev, 0, p*sizeof(T1));
        } else {
            for (size_t i = 0; i < p; ++i)
                w[i] = (T1) _param->w_initial[i];
            // w_prev is set in initialStep()
        }

        memset(D, 0, p*sizeof(T1));

        sols = new Solution<T1>(max_iter, p);
        work_set = new work_set_struct(p);
        lR = new LBFGS<T1>(p, l, (T1) param->shrink);

        ista_size = (T1) _param->stepsize_initial;

    };

    ~LHAC() {
        delete [] w_prev;
        delete [] w;
        delete [] L_grad;
        delete [] L_grad_prev;
        delete [] D;
        delete [] H_diag;
        delete [] d_bar;
        delete lR;
        delete work_set;

    }

    int ista() {
        T1 elapsedTimeBegin = CFAbsoluteTimeGetCurrent();
        int error = 0;
        normsg = normsg0;
        for (ista_iter = 1; ista_iter <= max_iter; ista_iter++) {
            error = istaStep();
            if (error) {
                break;
            }
            T1 elapsedTime = CFAbsoluteTimeGetCurrent()-elapsedTimeBegin;
            if (ista_iter == 1 || ista_iter % 30 == 0 )
                sols->addEntry(obj->val, normsg, elapsedTime, ista_iter, work_set->numActive);
            if (msgFlag >= LHAC_MSG_NEWTON)
                printf("%.4e  iter %3d:   obj.f = %+.4e    obj.normsg = %+.4e\n",
                       elapsedTime, ista_iter, obj->f, normsg);
            normsg = computeSubgradient();
            mdl->computeGradient(w, L_grad);
            if (normsg <= opt_outer_tol*normsg0) {
                break;
            }
        }
        return error;
    }

    // proximal inexact quasi-newton
    int piqn() {
        T1 elapsedTimeBegin = CFAbsoluteTimeGetCurrent();
        initialStep();
        int error = 0;
        for (newton_iter = 1; newton_iter < max_iter; newton_iter++) {
            computeWorkSet();
            lR->computeLowRankApprox_v2(work_set);
            T1 elapsedTime = CFAbsoluteTimeGetCurrent()-elapsedTimeBegin;
            normsg = computeSubgradient();
            if (msgFlag >= LHAC_MSG_NEWTON)
                printf("%.4e  iter %3d:   obj.f = %+.4e    obj.normsg = %+.4e   |work_set| = %ld\n",
                       elapsedTime, newton_iter, obj->f, normsg, work_set->numActive);
            sols->addEntry(obj->val, normsg, elapsedTime, newton_iter, work_set->numActive);
            if (normsg <= opt_outer_tol*normsg0) {
                break;
            }
            error = suffcientDecrease();
            if (error) {
                break;
            }
            memcpy(L_grad_prev, L_grad, p*sizeof(T1));
            mdl->computeGradient(w, L_grad);
            /* update LBFGS */
            lR->updateLBFGS(w, w_prev, L_grad, L_grad_prev);
        }
        return error;
    }

    template <typename InnerSolver>
    int piqnGeneral(Subproblem<InnerSolver, T1>* subprob) {
        double elapsedTimeBegin = CFAbsoluteTimeGetCurrent();
        initialStep();
        int error = 0;
        unsigned short max_inner_iter = 200;
        for (newton_iter = 1; newton_iter < max_iter; newton_iter++) {
            computeWorkSet();
            lR->computeLowRankApprox_v2(work_set);
            double elapsedTime = CFAbsoluteTimeGetCurrent()-elapsedTimeBegin;
            normsg = computeSubgradient();
            if (msgFlag >= LHAC_MSG_NEWTON)
                printf("%.4e  iter %3d:   obj.f = %+.4e    obj.normsg = %+.4e   |work_set| = %ld\n",
                       elapsedTime, newton_iter, obj->f, normsg, work_set->numActive);
            sols->addEntry(obj->val, normsg, elapsedTime, newton_iter, work_set->numActive);
            if (normsg <= opt_outer_tol*normsg0) {
                break;
            }

            /* inner solver starts*/
            subprob->build(lR, L_grad, work_set);
            T1 gama = lR->gama;
            T1 rho_trial = 0.0;
            memcpy(w_prev, w, p*sizeof(T1));
            unsigned short inner_iter;
            for (inner_iter = 0; inner_iter < max_inner_iter; inner_iter++) {
                const T1* d = subprob->solve(w_prev, newton_iter, gama);
                bool good_d = sufficientDecreaseCheck(d, subprob, gama, &rho_trial);
                if (good_d) {
                    if (msgFlag >= LHAC_MSG_SD)
                        printf("\t \t \t # of line searches = %3d; model quality: %+.3f\n", inner_iter, rho_trial);
                    break;
                }
                else
                    gama *= 2.0;

            }
            /* inner solver ends */
            if (inner_iter >= max_inner_iter) {
                error = 1;
                break;
            }

            memcpy(L_grad_prev, L_grad, p*sizeof(T1));
            mdl->computeGradient(w, L_grad);
            /* update LBFGS */
            lR->updateLBFGS(w, w_prev, L_grad, L_grad_prev);
        }
        return error;
    }

    /* fast proximal inexact quasi-newton */
    int fpiqn() {
        T1 elapsedTimeBegin = CFAbsoluteTimeGetCurrent();
        initialStep();
        T1 t = 1.0;
        int error = 0;
        T1* x = new T1[p];
        memcpy(x, w, p*sizeof(T1)); // w_1 (y_1) == x_0
        for (newton_iter = 1; newton_iter < max_iter; newton_iter++) {
            computeWorkSet();
            lR->computeLowRankApprox_v2(work_set);
            T1 elapsedTime = CFAbsoluteTimeGetCurrent()-elapsedTimeBegin;
            normsg = computeSubgradient();
            if (msgFlag >= LHAC_MSG_NEWTON)
                printf("%.4e  iter %3d:   obj.f = %+.4e    obj.normsg = %+.4e   |work_set| = %ld\n",
                       elapsedTime, newton_iter, obj->f, normsg, work_set->numActive);
            sols->addEntry(obj->val, normsg, elapsedTime, newton_iter, work_set->numActive);
            if (normsg <= opt_outer_tol*normsg0) {
                break;
            }
            error = suffcientDecrease();
            if (error) {
                break;
            }
            fistaUpdate(&t, x);
            obj->add(mdl->computeObject(w), computeReg(w));
            memcpy(L_grad_prev, L_grad, p*sizeof(T1));
            mdl->computeGradient(w, L_grad);
            /* update LBFGS */
            lR->updateLBFGS(w, w_prev, L_grad, L_grad_prev);
        }
        return error;
    }

    Solution<T1>* solve()
    {
        obj->add(mdl->computeObject(w), computeReg(w));
        mdl->computeGradient(w, L_grad);
        normsg0 = computeSubgradient();
        int error = 0;

        switch (param->method_flag) {
            case 1:
                error = ista();
                break;

            case 2:
                error = piqn();
                break;

            case 3:
                error = fpiqn();
                break;

            case 4:
                error = piqnGeneral(new CoordinateDescent<T1>(param, p));
                break;

            default:
                error = 1;
                fprintf(stderr, "ValueError: flag q only accept value 1 (ISTA), 2 (lhac) or 3 (f-lhac).\n");
                break;
        }

        sols->finalReport(error, w);
        return sols;
    };

private:
    Objective<Derived, T1>* mdl;
    const Parameter* param;
    Solution<T1>* sols;
    work_set_struct* work_set;
    Func* obj;
    LBFGS<T1>* lR;

    unsigned long l;
    T1 opt_outer_tol;
    unsigned short max_iter;
    T1 lmd;
    int msgFlag;

    unsigned long p;
    unsigned short newton_iter;
    unsigned short ista_iter;
    T1 ista_size;

    T1* D;
    T1 normsg0;
    T1 normsg;
    T1* w_prev;
    T1* w;
    T1* L_grad_prev;
    T1* L_grad;
    T1* H_diag; // p
    T1* d_bar; // 2*l


    void initialStep() {
        // initial step (only for l1)
        // for (unsigned long idx = 0; idx < p; idx++) {
        //     T1 G = L_grad[idx];
        //     T1 Gp = G + lmd;
        //     T1 Gn = G - lmd;
        //     T1 Hwd = 0.0;
        //     if (Gp <= Hwd)
        //         D[idx] = -Gp;
        //     else if (Gn >= Hwd)
        //         D[idx] = -Gn;
        //     else
        //         D[idx] = 0.0;
        // }
        // T1 a = 1.0;
        // T1 l1_next = 0.0;
        // T1 delta = 0.0;
        // for (unsigned long i = 0; i < p; i++) {
        //     w[i] += D[i];
        //     l1_next += lmd*fabs(w[i]);
        //     delta += L_grad[i]*D[i];
        // }
        // delta += l1_next - obj->g;
        // // line search
        // for (unsigned long lineiter = 0; lineiter < 1000; lineiter++) {
        //     T1 f_trial = mdl->computeObject(w);
        //     T1 obj_trial = f_trial + l1_next;
        //     if (obj_trial < obj->val + a*0.001*delta) {
        //         obj->add(f_trial, l1_next);
        //         break;
        //     }
        //     a = 0.5*a;
        //     l1_next = 0;
        //     for (unsigned long i = 0; i < p; i++) {
        //         w[i] = w_prev[i] + a*D[i];
        //         l1_next += lmd*fabs(w[i]);
        //     }
        // }
        istaStep();
        memcpy(L_grad_prev, L_grad, p*sizeof(T1));
        mdl->computeGradient(w, L_grad);
        lR->initData(w, w_prev, L_grad, L_grad_prev);
    }

    int istaStep() {
        printf("Finding the proper initial step size...\n");
        memcpy(w_prev, w, p*sizeof(T1));
        for (int backtrack=0; backtrack<200; backtrack++) {
            T1 t = ista_size*lmd;
            unsigned long i;
#pragma omp parallel for private(i)
            for (i = 0; i < p; i++) {
                T1 ui = w_prev[i] - ista_size*L_grad[i];
                if (ui > t)
                    w[i] = ui - t;
                else if (ui < -t)
                    w[i] = ui + t;
                else
                    w[i] = 0.0;
                D[i] = w[i] - w_prev[i];
            }
            T1 order1 = lcddot((int)p, D, 1, L_grad, 1);
            T1 order2 = lcddot((int)p, D, 1, D, 1);
            T1 f_trial = mdl->computeObject(w);
            if (f_trial > obj->f + order1 + (0.5/ista_size)*order2) {
                ista_size = ista_size * 0.5;
                continue;
            }
            printf("SET initial step size to %f\n", ista_size);
            obj->add(f_trial, 0);
            return 0;
        }
        return 1;
    }

    void fistaUpdate(T1* const t, T1* const x) {
        T1 t_ = *t;
        *t = (1 + sqrt(1+4*t_*t_))*0.5;
        T1 c = (t_ - 1) / *t;
        for (unsigned long i = 0; i < p; i++) {
            T1 yi = w[i] + c*(w[i] - x[i]); // x is x_{k-1}
            x[i] = w[i];
            w[i] = yi;
        }
    }

    /* may generalize to other regularizations beyond l1 */
    T1 computeReg(const T1* const wnew) {
        T1 gval = 0.0;
        for (unsigned long i = 0; i < p; i++)
            gval += lmd*fabs(wnew[i]);
        return gval;
    }

    T1 computeSubgradient() {
        T1 subgrad = 0.0;
        for (unsigned long i = 0; i < p; i++) {
            T1 g = L_grad[i];
            if (w[i] != 0.0 || (fabs(g) > lmd)) {
                if (w[i] > 0)
                    g += lmd;
                else if (w[i] < 0)
                    g -= lmd;
                else
                    g = fabs(g) - lmd;
                subgrad += fabs(g);
            }
        }
        return subgrad;
    }

    static int _cmp_by_vlt(const void *a, const void *b)
    {
        const ushort_pair_t *ia = (ushort_pair_t *)a;
        const ushort_pair_t *ib = (ushort_pair_t *)b;

        if (ib->vlt - ia->vlt > 0) {
            return 1;
        }
        else if (ib->vlt - ia->vlt < 0){
            return -1;
        }
        else
            return 0;
    }

    static int _cmp_by_vlt_reverse(const void *a, const void *b)
    {
        const ushort_pair_t *ia = (ushort_pair_t *)a;
        const ushort_pair_t *ib = (ushort_pair_t *)b;
        if (ib->vlt - ia->vlt > 0) {
            return -1;
        }
        else if (ib->vlt - ia->vlt < 0){
            return 1;
        }
        else
            return 0;
    }

    void computeWorkSet()
    {
        switch (param->active_set) {
            case GREEDY:
                greedySelector();
                break;

            case STD:
                stdSelector();
                break;

            case STD_CUTGRAD:
                stdSelector_cutgrad();
                break;

            case STD_CUTGRAD_AGGRESSIVE:
                stdSelector_cutgrad_aggressive();
                break;

            case GREEDY_CUTGRAD:
                greedySelector();
                break;

            case GREEDY_CUTZERO:
                greedySelector_cutzero();
                break;

            case GREEDY_ADDZERO:
                greedySelector_addzero();
                break;

            default:
                stdSelector();
                break;
        }
        /* reset permutation */
        for (unsigned long j = 0; j < work_set->numActive; j++) {
            work_set->permut[j] = j;
        }
        return;
    }

    void stdSelector()
    {

        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long numActive = 0;
        /*** select rule 2 ***/
        for (unsigned long j = 0; j < p; j++) {
            T1 g = L_grad[j];
            if (w[j] != 0.0 || (fabs(g) > lmd)) {
                idxs[numActive].i = (unsigned short) j;
                idxs[numActive].j = (unsigned short) j;
                numActive++;
            }
        }
        work_set->numActive = numActive;
        return;
    }

    void stdSelector_cutgrad()
    {
        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long numActive = 0;
        /*** select rule 2 ***/
        for (unsigned long j = 0; j < p; j++) {
            T1 g = L_grad[j];
            if (w[j] != 0.0 || (fabs(g) > lmd + 0.01)) {
                idxs[numActive].i = (unsigned short) j;
                idxs[numActive].j = (unsigned short) j;
                numActive++;
            }
        }
        work_set->numActive = numActive;
        return;
    }

    void stdSelector_cutgrad_aggressive()
    {
        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long numActive = 0;
        /*** select rule 2 ***/
        for (unsigned long j = 0; j < p; j++) {
            T1 g = L_grad[j];
            if (w[j] != 0.0 || (fabs(g) > lmd + 0.5)) {
                idxs[numActive].i = (unsigned short) j;
                idxs[numActive].j = (unsigned short) j;
                numActive++;
            }
        }
        work_set->numActive = numActive;
        return;
    }

    void greedySelector()
    {
        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long numActive = 0;
        unsigned long zeroActive = 0;
        for (unsigned long j = 0; j < p; j++) {
            T1 g = L_grad[j];
            if (w[j] != 0.0 || (fabs(g) > lmd)) {
                idxs[numActive].i = (unsigned short) j;
                idxs[numActive].j = (unsigned short) j;
                g = fabs(g) - lmd;
                idxs[numActive].vlt = fabs(g);
                numActive++;
                if (w[j] == 0.0) zeroActive++;
            }
        }
        qsort((void *)idxs, (size_t) numActive, sizeof(ushort_pair_t), _cmp_by_vlt);
        work_set->numActive = numActive;
    }


    void greedySelector_cutgrad()
    {
        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long numActive = 0;
        unsigned long zeroActive = 0;
        for (unsigned long j = 0; j < p; j++) {
            T1 g = L_grad[j];
            if (w[j] != 0.0 || (fabs(g) > lmd + 0.01)) {
                idxs[numActive].i = (unsigned short) j;
                idxs[numActive].j = (unsigned short) j;
                g = fabs(g) - lmd;
                idxs[numActive].vlt = fabs(g);
                numActive++;
                if (w[j] == 0.0) zeroActive++;
            }
        }
        qsort((void *)idxs, (size_t) numActive, sizeof(ushort_pair_t), _cmp_by_vlt);
        work_set->numActive = numActive;
    }

    void greedySelector_cutzero()
    {
        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long numActive = 0;
        unsigned long zeroActive = 0;
        for (unsigned long j = 0; j < p; j++) {
            T1 g = L_grad[j];
            if (w[j] != 0.0 || (fabs(g) > lmd)) {
                idxs[numActive].i = j;
                idxs[numActive].j = j;
                g = fabs(g) - lmd;
                idxs[numActive].vlt = fabs(g);
                numActive++;
                if (w[j] == 0.0) zeroActive++;
            }
        }
        qsort((void *)idxs, (size_t) numActive, sizeof(ushort_pair_t), _cmp_by_vlt);
        // zerosActive small means found the nonzeros subspace
        numActive = (zeroActive<100)?numActive:(numActive-zeroActive);
        work_set->numActive = numActive;
    }

    void _insert(unsigned long idx, T1 vlt, unsigned long n)
    {
        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long end = p-1-n;
        unsigned long j;
        for (j = p-1; j > end; j--) {
            if (idxs[j].vlt >= vlt) continue;
            else {
                for (unsigned long i = j+1, k = j; i > end; i--, k--) {
                    // swap
                    unsigned long tmpj = idxs[k].j;
                    T1 tmpv = idxs[k].vlt;
                    idxs[k].j = idx;
                    idxs[k].vlt = vlt;
                    vlt = tmpv;
                    idx = tmpj;
                }
                break;
            }
        }
        if (j == end) {
            idxs[end].j = idx;
            idxs[end].vlt = vlt;
        }
    }

    T1 _vlt(unsigned long j)
    {
        T1 g = L_grad[j];
        if (w[j] > 0) g += lmd;
        else if (w[j] < 0) g -= lmd;
        else g = fabs(g) - lmd;
        return g;
    }


    /* not converging on a9a */
    void greedySelector_addzero_no()
    {
        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long numActive = 0;
        unsigned long work_size = param->work_size;
        unsigned long zeroActive = 0;
        unsigned long nzeroActive = 0;
        for (unsigned long j = 0; j < p; j++) {
            T1 g = fabs(L_grad[j]) - lmd;
            if (g > 0) {
                unsigned long end = p-1-zeroActive;
                idxs[end].j = j;
                idxs[end].vlt = g;
                zeroActive++;
            }
            else if (w[j] != 0.0) {
                idxs[nzeroActive].j = j;
                nzeroActive++;
            }
        }
        if (zeroActive>2*nzeroActive) {
            unsigned long pos = p - zeroActive;
            qsort((void *)(idxs+pos), (size_t) zeroActive, sizeof(ushort_pair_t), _cmp_by_vlt_reverse);
            work_size = (nzeroActive<10)?zeroActive/3:nzeroActive;
        }
        else work_size = zeroActive;
        numActive = nzeroActive;
        unsigned long end = p-work_size;
        for (unsigned long j = p-1; j >= end; j--) {
            idxs[numActive].j = idxs[j].j;
            numActive++;
        }
        work_set->numActive = numActive;
    }

    void greedySelector_addzero()
    {
        ushort_pair_t* &idxs = work_set->idxs;
        unsigned long numActive = 0;
        unsigned long work_size = param->work_size;
        unsigned long zeroActive = 0;
        unsigned long nzeroActive = 0;
        for (unsigned long j = 0; j < p; j++) {
            T1 g = fabs(L_grad[j]) - lmd;
            if (g > 0) {
                _insert(j, g, zeroActive);
                zeroActive++;
            }
            else if (w[j] != 0.0) {
                idxs[nzeroActive].j = j;
                nzeroActive++;
            }
        }
        work_size = (nzeroActive<10)?zeroActive:nzeroActive;
        work_size = (zeroActive>2*nzeroActive)?work_size:zeroActive;
        numActive = nzeroActive;
        unsigned long end = p-work_size;
        for (unsigned long k = p, j = p-1; k > end; k--, j--) {
            idxs[numActive].j = idxs[j].j;
            numActive++;
        }
        work_set->numActive = numActive;
    }

    static inline void shuffle( work_set_struct* work_set )
    {
        unsigned long lens = work_set->numActive;
        ushort_pair_t* idxs = work_set->idxs;
        unsigned long* permut = work_set->permut;

        for (unsigned long i = 0; i < lens; i++) {
            unsigned long j = i + rand()%(lens - i);
            unsigned short k1 = idxs[i].i;
            unsigned short k2 = idxs[i].j;
            T1 vlt = idxs[i].vlt;
            idxs[i].i = idxs[j].i;
            idxs[i].j = idxs[j].j;
            idxs[i].vlt = idxs[j].vlt;
            idxs[j].i = k1;
            idxs[j].j = k2;
            idxs[j].vlt = vlt;

            /* update permutation */
            unsigned long tmp = permut[i];
            permut[i] = permut[j];
            permut[j] = tmp;
        }

        return;
    }


    int suffcientDecrease() {
        int max_sd_iters = 200;
        T1 mu = 1.0;
        T1 rho = param->rho;
        int msgFlag = param->verbose;
        memcpy(w_prev, w, p*sizeof(T1));
        const T1 lmd = param->lmd;
        const unsigned long l = param->l;
        T1* Q = lR->Q;
        const T1* Q_bar = lR->Q_bar;
        const unsigned short m = lR->m;
        const T1 gama = lR->gama;
        memset(D, 0, p*sizeof(T1));
        memset(d_bar, 0, 2*l*sizeof(T1));
        for (unsigned long k = 0, i = 0; i < work_set->numActive; i++, k += m) {
            H_diag[i] = gama;
            for (unsigned long j = 0; j < m; j++) H_diag[i] -= Q_bar[k+j]*Q[k+j];
        }
        unsigned long max_cd_pass = 1 + newton_iter / param->cd_rate;
        unsigned long* permut = work_set->permut;
        ushort_pair_t* idxs = work_set->idxs;
        unsigned long cd_pass;
        int sd_iters;
        for (sd_iters = 0; sd_iters < max_sd_iters; sd_iters++) {
            T1 gama_scale = mu*gama;
            T1 dH_diag = gama_scale-gama;
            for (cd_pass = 1; cd_pass <= max_cd_pass; cd_pass++) {
                T1 diffd = 0;
                T1 normd = 0;
                for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
                    unsigned long rii = ii;
                    unsigned long idx = idxs[rii].j;
                    unsigned long idx_Q = permut[rii];
                    unsigned long Q_idx_m = idx_Q*m;
                    T1 Qd_bar = lcddot(m, &Q[Q_idx_m], 1, d_bar, 1);
                    T1 Hd_j = gama_scale*D[idx] - Qd_bar;
                    T1 Hii = H_diag[idx_Q] + dH_diag;
                    T1 G = Hd_j + L_grad[idx];
                    T1 Gp = G + lmd;
                    T1 Gn = G - lmd;
                    T1 wpd = w_prev[idx] + D[idx];
                    T1 Hwd = Hii * wpd;
                    T1 z = -wpd;
                    if (Gp <= Hwd) z = -Gp/Hii;
                    if (Gn >= Hwd) z = -Gn/Hii;
                    D[idx] = D[idx] + z;
                    for (unsigned long k = Q_idx_m, j = 0; j < m; j++)
                        d_bar[j] += z*Q_bar[k+j];
                    diffd += fabs(z);
                    normd += fabs(D[idx]);
                }
                if (msgFlag >= LHAC_MSG_CD) {
                    printf("\t\t Coordinate descent pass %ld:   Change in d = %+.4e   norm(d) = %+.4e\n",
                           cd_pass, diffd, normd);
                }
            }
            for (unsigned long i = 0; i < p; i++) {
                w[i] = w_prev[i] + D[i];
            }
            T1 f_trial = mdl->computeObject(w);
            T1 g_trial = computeReg(w);
            T1 obj_trial = f_trial + g_trial;
            T1 order1 = lcddot((int)p, D, 1, L_grad, 1);
            T1 order2 = 0;
            T1* buffer = lR->buff;
            int cblas_M = (int) work_set->numActive;
            int cblas_N = (int) m;
            lcdgemv(CblasColMajor, CblasTrans, Q, d_bar, buffer, cblas_N, cblas_M, cblas_N);
            T1 vp = 0;
            for (unsigned long ii = 0; ii < work_set->numActive; ii++) {
                unsigned long idx = idxs[ii].j;
                unsigned long idx_Q = permut[ii];
                vp += D[idx]*buffer[idx_Q];
            }
            order2 = mu*gama*lcddot((int)p, D, 1, D, 1)-vp;
            order2 = order2*0.5;
            T1 f_mdl = obj->f + order1 + order2 + g_trial;
            T1 rho_trial = (obj_trial-obj->val)/(f_mdl-obj->val);
            if (msgFlag >= LHAC_MSG_SD) {
                printf("\t \t \t # of line searches = %3d; model quality: %+.3f\n", sd_iters, rho_trial);
            }
            if (rho_trial > rho) {
                obj->add(f_trial, g_trial);
                break;
            }
            mu = 2*mu;
        }
        if (sd_iters == max_sd_iters) {
            fprintf(stderr, "failed to satisfy sufficient decrease condition.\n");
            return -1;
        }
        return 0;
    }

    template <typename InnerSolver>
    bool sufficientDecreaseCheck(const T1* D, Subproblem<InnerSolver, T1>* const subprob,
                                 const T1 gama, T1* rho_trial) {
        for (unsigned long i = 0; i < p; i++) {
            w[i] = w_prev[i] + D[i];
        }
        T1 f_trial = mdl->computeObject(w);
        T1 g_trial = computeReg(w);
        T1 obj_trial = f_trial + g_trial;
        T1 f_mdl = obj->f + subprob->objective_value(gama) + g_trial;
        *rho_trial = (obj_trial-obj->val)/(f_mdl-obj->val);
        if (*rho_trial > param->rho) {
            obj->add(f_trial, g_trial);
            return true;
        }
        return false;

    }
};






#endif /* defined(__LHAC_v1__lhac__) */
