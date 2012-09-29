//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// Copyright (C) 2012 Computer Graphics Systems Group at the
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

class FFTGkfTest : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("category", "Kuwahara Filter");
public:
    Q_INVOKABLE FFTGkfTest();
    virtual void process();

public slots:
    void benchmark();

public:
    int N_;
    int k_;
    double radius_;
    double precision_;
    double smoothing_;
    double q_;
    double threshold_;
    int benchmark_n;
};
