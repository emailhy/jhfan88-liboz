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

class ShockFDoGUpwind : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("category", "PDE");
public:
    Q_INVOKABLE ShockFDoGUpwind();
    virtual void process();

    int N;
    double rho;
    double sigma;
    double fdog_sigma;
    double fdog_tau ;
    double step;
};


class WeickertCohEnhShock : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Coherence-Enhancing Shock Filter\n[Weickert, 2003]")
    Q_CLASSINFO("category", "PDE");
public:
    Q_INVOKABLE WeickertCohEnhShock();
    virtual void process();

    int N;
    double rho;
    QString smooting;
    double alpha;
    double sigma;
    double step;
};


class OsherRudinShock : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Feature-Oriented Image enhancement using Shock Filters\n[Osher & Rudin, 1990]")
    Q_CLASSINFO("category", "PDE");
public:
    Q_INVOKABLE OsherRudinShock();
    virtual void process();

    int N;
    double dt;
};


class AlvarezMazorraShock : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Signal and image restoration using shock filters and anisotropic diffusion\n[Alvarez & Mazorra, 1994]")
    Q_CLASSINFO("category", "PDE");
public:
    Q_INVOKABLE AlvarezMazorraShock();
    virtual void process();

    bool pre_blur;
    int N;
    double c;
    double sigma;
    double dt;
};
