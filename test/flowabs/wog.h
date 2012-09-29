//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2010-2012 Computer Graphics Systems Group at the
// Hasso-Plattner-Institut, Potsdam, Germany <www.hpi3d.de>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
#pragma once

#include "simplemodule.h"

class WOG : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Real-Time Video Abstraction\n[Winnemöller et al., 2006]");
    Q_CLASSINFO("category", "Bilateral Filter");
    Q_CLASSINFO("rating", "***");
public:
    Q_INVOKABLE WOG();
    virtual void process();

    QString filter_type;
    int n_a;
    int n_e;
    double sigma_d;
    double sigma_r;
    bool dog;
    double sigma_e;
    double precision_e;
    double tau;
    double phi_e;
    double epsilon;
    bool quantization;
    QString quant_type;
    int nbins;
    double phi_q;
    double lambda_delta;
    double omega_delta;
    double lambda_phi;
    double omega_phi;
    QString background;
    bool warp_sharp;
    double sigma_w;
    double precision_w;
    double phi_w;
};
