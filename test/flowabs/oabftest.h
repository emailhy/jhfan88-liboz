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
#include <oz/oabf.h>

class OaBfTest : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Orientation-aligned Bilateral Filter\n[Kyprianidis and Döllner, 2008]")
    Q_CLASSINFO("category", "Bilateral Filter");
public:
    Q_INVOKABLE OaBfTest();
    virtual void process();
    virtual void draw(ImageView *view, QPainter &p, int pass);
    virtual void dragBegin(ImageView *view, QMouseEvent* e);

    bool noise;
    double variance;
    double noise_variance;
    oz::gpu_image noise_img;

    int N;
    QString method;
    QString st_colorspace;
    QString bf_colorspace;
    QString order;
    double rho;
    bool st_recalc;
    double sigma_dg;
    double sigma_rg;
    double precision_g;
    double sigma_dt;
    double sigma_rt;
    double precision_t;
    bool adaptive;
    bool ustep;
    QString src_linear_ex;
    bool st_linear;
    double step_size;
    bool nonuniform_sigma;

    bool quantization;
    QString quant_type;
    int nbins;
    double phi_q;
    double lambda_delta;
    double omega_delta;
    double lambda_phi;
    double omega_phi;

    bool debug;
    double debug_x;
    double debug_y;
    bool draw_orientation;
    bool draw_center;
    bool draw_line_p1;
    bool draw_active_p1;
    bool draw_samples_p1;
    bool draw_line_p2;
    bool draw_active_p2;
    bool draw_samples_p2;
    bool draw_midpoint;
    bool show_plot;

    ParamDouble *p_debug_x;
    ParamDouble *p_debug_y;
};
