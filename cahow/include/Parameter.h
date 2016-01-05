//
//  Parameter.h
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 10/18/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#ifndef LHAC_v1_Parameter_h
#define LHAC_v1_Parameter_h
#include <stdio.h>

struct Parameter {
    unsigned long work_size;
    unsigned short max_iter;
    double lmd;
    double opt_outer_tol;
    unsigned long l; // lbfgs sy pair number <= MAX_SY_PAIRS
    int verbose;

    /* gama in lbfgs */
    double shrink; // gama = gama/shrink
    double rho;
    unsigned long cd_rate;

    // active set stragety
    unsigned long active_set;
    int method_flag;
};

#endif
