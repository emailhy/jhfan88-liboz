// by Jan Eric Kyprianidis <www.kyprianidis.com>
#pragma once

#include "simplemodule.h"

class FFTConvTest : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("category", "Test");
public:
    Q_INVOKABLE FFTConvTest();
    virtual void process();

    int N_;
    int k_;
    double sigma_r_;
};
