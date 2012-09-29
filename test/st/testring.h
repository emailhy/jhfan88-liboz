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

class TestRing : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("category", "Test");
    Q_CLASSINFO("rating", "debug");
public:
    Q_INVOKABLE TestRing();
    virtual void process();
    void process2(const QString& prefix, const oz::gpu_image& I, const oz::gpu_image& mask);
    virtual void dragBegin(ImageView *view, QMouseEvent *e);
    virtual void draw(ImageView *view, QPainter &p, int pass);
    void mergeHalfs(const QString& c, const QString& a, const QString& b);
    void mergeQuads(const QString& r, const QString& a, const QString& b, const QString& c, const QString& d);

    QString image_type;
    bool image_8bit;
    int width;
    int height;
    bool sRGB;
    double g0;
    double km;
    double rm;
    double w;
    bool inverted;
    bool equalize;
    double variance;
    QString gradient;
    int pyrdownMode;
    double pre_blur;
    double sigma;
    double precision_sigma;
    double rho;
    double rho8;
    double precision_rho;
    double m;
    bool st_normalize;
    int etf_N;
    int moa_mode;
    bool draw_scale;
    bool draw_st;
    double scale;
    bool ds_squared;
    bool st_8bit;
    double st_scale;
    double st_gamma;
    QString st_resample;
    double hist_max;
    QString st_up;
    QString st_down;
    bool st_liegr;
    oz::gpu_image noise;
    double noise_var;
    bool use_grad;
    double grad_scale;
};

oz::gpu_image testring_diff_angle( const oz::gpu_image& angle );
oz::gpu_image testring_xy_angle( int w, int h );
oz::gpu_image testring_jet( const oz::gpu_image& diff, float scale );
