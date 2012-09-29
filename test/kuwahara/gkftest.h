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

class GeneralizedKuwahara : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Artistic Edge and Corner Preserving Smoothing\n[Papari et al., 2007]");
    Q_CLASSINFO("category", "Kuwahara Filter");
public:
    Q_INVOKABLE GeneralizedKuwahara();
    virtual void process();

    QString method;
    int N;
    double smoothing;
    double precision;
    int radius;
    double q;
    oz::gpu_image krnl41;
    oz::gpu_image krnl81;
    oz::gpu_image krnl84;
    oz::gpu_image krnl8x2;
    double krnl_smoothing;
    double krnl_precision;
    double threshold;
};
