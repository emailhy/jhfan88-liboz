// by Jan Eric Kyprianidis <www.kyprianidis.com>
#pragma once

#include "simplemodule.h"

class VarianceTest : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("category", "Test");
public:
    Q_INVOKABLE VarianceTest();
    virtual void process();

protected:
    QString type;
    int w, h;
    double mean, variance;
};

