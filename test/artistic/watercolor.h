// by Jan Eric Kyprianidis <www.kyprianidis.com>
#pragma once

#include "simplemodule.h"

class Watercolor : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("category", "Artistic");
    Q_CLASSINFO("description", "Simple Watercolor similar to NPRC implementation");
public:
    Q_INVOKABLE Watercolor();
    virtual void process();

    double bf_sigma_d;
    double bf_sigma_r;
    int bf_N;
    int nbins;
    double phi_q;
    double sigma_c;
    double nalpha;
    double nbeta;
};
