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


class SSIA : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Shape-simplifying Image Abstraction\n[Kang & Lee, 2008]")
    Q_CLASSINFO("category", "EG2011");
    Q_CLASSINFO("rating", "***");
public:
    Q_INVOKABLE SSIA();
    virtual void process();

    double pre_smooth;
    int total_N;
    int cmcf_N;
    double cmcf_weight;
    int etf_N;
    int etf_halfw;
    double shock_sigma;
    double shock_tau;
    double final_smooth;
};


class SSIA2 : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Shape-simplifying Image Abstraction (Modified)")
    Q_CLASSINFO("category", "EG2011");
public:
    Q_INVOKABLE SSIA2();
    virtual void process();

    double pre_smooth;
    int total_N;
    QString flow_type;
    double st_sigma;
    int etf_N;
    int etf_halfw;
    QString cmcf_type;
    int cmcf_N;
    double cmcf_step;
    double cmcf_weight;
    QString dog_type;
    QString shock_type;
    double shock_step;
    double shock_sigma;
    double shock_tau;
    double final_smooth;
};
