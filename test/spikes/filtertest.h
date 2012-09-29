// by Jan Eric Kyprianidis <www.kyprianidis.com>
#pragma once

#include "simplemodule.h"

class FilterTest : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("category", "Test");
public:
    Q_INVOKABLE FilterTest();
    virtual void process();

protected:
    QString type;
    double rho;
    double sigma;
    double alpha;
    double angle;
};
