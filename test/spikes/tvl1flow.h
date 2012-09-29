// by Jan Eric Kyprianidis <www.kyprianidis.com>
#pragma once

#include "simplemodule.h"

class TVL1Flow : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("category", "Test");
public:
    Q_INVOKABLE TVL1Flow();
    virtual void process();

protected:
    double pre_smooth;
    double pyr_scale;
    int warps;
    int maxits;
    double lambda;
    bool warp_noise;
    double max_radius;
};
