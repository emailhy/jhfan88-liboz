// by Jan Eric Kyprianidis <www.kyprianidis.com>
#pragma once

#include "simplemodule.h"
#include <oz/laplace_eq.h>

class LaplaceEqTest : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("category", "Test");
public:
    Q_INVOKABLE LaplaceEqTest();
    virtual void process();
    oz::gpu_image vcycle( const oz::gpu_image b, int level );

protected:
    QString type;
    oz::leq_stencil_t stencil;
    oz::leq_upfilt_t upfilt;
    int v2;
    double err_scale;
    std::vector<oz::gpu_image> P;
    std::vector<oz::gpu_image> Q;
    std::vector<oz::gpu_image> R;
    oz::gpu_image ref;
};

