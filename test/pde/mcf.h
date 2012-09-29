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

class MCF : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Proposed method")
    Q_CLASSINFO("category", "PDE");
public:
    Q_INVOKABLE MCF();
    virtual void process();

    int N;
    double sigma;
    double step;
    QString type;
    double epsilon;
    double lambda;
    double p;

    /*
    double pre_smooth;
    int cmcf_N;
    double cmcf_weight;
    int etf_N;
    int etf_halfw;
    double shock_sigma;
    double shock_tau;
    double final_smooth;
    */
};
