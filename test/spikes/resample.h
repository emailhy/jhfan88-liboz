// by Jan Eric Kyprianidis <www.kyprianidis.com>
#pragma once

#include "simplemodule.h"

class ResampleTest : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("category", "Test");
public:
    Q_INVOKABLE ResampleTest();
    virtual void process();

protected:
    QString image_type;
    int zwidth;
    int zheight;
    double g0;
    double rm;
    double km;
    double w;
    bool inverted;
    bool sRGB;

    int mode;
    int rwidth, rheight;
};

