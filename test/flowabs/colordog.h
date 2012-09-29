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

class ColorDog : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Proposed Method")
    Q_CLASSINFO("description", "Coherent Line Drawings of Color Images")
    Q_CLASSINFO("category", "Bilateral Filter")
    Q_CLASSINFO("rating", "debug");
public:
    Q_INVOKABLE ColorDog();
    virtual void process();

    QString st_type;
    double rho;
    double tau_r;

    QString pre_smooth;
    int N;
    double sigma_g;
    double sigma_t;
    double sigma_r;
    double max_angle;
    double precision;

    QString dog_type;
    QString luminance;
    double sigma_e;
    double precision_e;
    double k;
    double sigma_m;
    double epsilon;
    double tau;
    double phi_e;
    double sigma_f;
};
