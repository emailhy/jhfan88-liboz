// by Jan Eric Kyprianidis <www.kyprianidis.com>
#pragma once

#include "simplemodule.h"

class ReductionTest : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("category", "Test");
public:
    Q_INVOKABLE ReductionTest();
    virtual void process();

protected:
    int threads;
    int blocks;
};

