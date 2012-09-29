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


class EdgeEnhancingDiff : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Edge-Enhancing Diffusion\n[Weickert, 1996]")
    Q_CLASSINFO("category", "PDE");
public:
    Q_INVOKABLE EdgeEnhancingDiff();
    virtual void process();

    int max_width;
    int max_height;
    bool grayscale;
    double sigma;
    double rho;
    double lambda;
    double dt;
    int N;
};


class CoherenceEnhancingDiff : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Coherence-Enhancing Diffusion\n[Weickert, 1996]")
    Q_CLASSINFO("category", "PDE");
public:
    Q_INVOKABLE CoherenceEnhancingDiff();
    virtual void process();

    int max_width;
    int max_height;
    bool grayscale;
    double sigma;
    double rho;
    double alpha;
    double C;
    int m;
    double dt;
    int N;
};
