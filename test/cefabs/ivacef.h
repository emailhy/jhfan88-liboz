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
#include <oz/laplace_eq.h>

class IVACEF : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Image and Video Abstraction by Coherence-Enhancing Filtering\n[Kyprianidis & Kang, 2011]")
    Q_CLASSINFO("description", "Eurographics 2011 paper reference implementation")
    Q_CLASSINFO("category", "EG2011")
    Q_CLASSINFO("rating", "***");
public:
    Q_INVOKABLE IVACEF();
    virtual void process();
    oz::gpu_image computeSt( const oz::gpu_image& src, const oz::gpu_image& prev );
    virtual void draw(ImageView *view, QPainter &p, int pass);
    virtual void dragBegin(ImageView *view, QMouseEvent* e);

    QPointF pos;
    int N;
    bool auto_levels;
    double auto_levels_threshold;
    bool noise;
    double variance;
    double noise_variance;
    oz::gpu_image noise_img;
    QString method;
    double sigma_d;
    double tau_r;
    oz::leq_stencil_t stencil;
    oz::leq_upfilt_t upfilt;
    int v2;
    double sigma_t;
    QString order;
    double step_size;
    bool adaptive;
    bool src_linear;
    bool st_linear;
    bool ustep;
    double sigma_i;
    double sigma_g;
    double r;
    double tau_s;
    double sigma_a;
    bool shock_filtering;

    bool debug;
    double debug_x;
    double debug_y;
    bool draw_gradients;
    bool draw_orientation;
    bool draw_streamline;
    bool draw_streamline_linear;
    bool show_plot;
    bool draw_midpoint;
    ParamDouble *p_debug_x;
    ParamDouble *p_debug_y;
};
