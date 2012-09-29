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

class TestPattern : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("category", "Test");
    Q_CLASSINFO("rating", "debug");
public:
    Q_INVOKABLE TestPattern();
    virtual void process();
    virtual void draw(ImageView *view, QPainter &p, int pass);

    QString pattern;
    int function;
    int width;
    int height;
    bool sRGB;
    bool noise;
    double variance;
    oz::gpu_image nimg;
    double nimg_variance;

    QString gradient;
    double rho;

    double phi;
    double phase;
    double scale;

    bool draw_st;
    bool draw_gf;
    bool draw_tf;
    bool draw_axis;
    bool draw_angle;
    bool draw_gradient;
    bool draw_tangent;
    bool draw_mini;
    bool draw_scale;
};

