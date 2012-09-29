// by Jan Eric Kyprianidis <www.kyprianidis.com>
#pragma once

#include "simplemodule.h"

class NoiseTest : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("category", "Test");
public:
    Q_INVOKABLE NoiseTest();
    virtual void process();

protected:
    QString type;
    int w, h;
    double mean, variance;
    double scale;
    bool normalize;
    double density;
};

