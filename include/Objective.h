//
//  Objective.h
//  LHAC_v1
//
//  Created by Xiaocheng Tang on 4/29/14.
//  Copyright (c) 2014 Xiaocheng Tang. All rights reserved.
//

#ifndef __LHAC_v1__Objective__
#define __LHAC_v1__Objective__


template <typename Derived, typename T1>
class Objective
{
public:
    inline unsigned long getDims() {
        return static_cast<Derived*>(this)->getDims();
    };

    inline T1 computeObject(T1* wnew) {
        return static_cast<Derived*>(this)->computeObject(wnew);
    };

    inline void computeGradient(T1* wnew, T1* df) {
        static_cast<Derived*>(this)->computeGradient(wnew, df);
    };

    virtual ~Objective() {};

};

#endif /* defined(__LHAC_v1__Objective__) */
