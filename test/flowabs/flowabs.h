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

class FlowAbs : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Image Abstraction by Structure Adaptive Filtering\n[Kyprianidis and Döllner, 2006]");
    Q_CLASSINFO("category", "Bilateral Filter");
    Q_CLASSINFO("rating", "***");
public:
    Q_INVOKABLE FlowAbs();
    virtual void process();
    virtual void draw(ImageView *view, QPainter &p, int pass);

    QString output;
    QString input_gamma;
    int max_width;
    int max_height;
    bool auto_levels;
    double auto_levels_threshold;

    bool noise;
    double variance;
    QString st_type;
    double rho;
    double precision_rho;
    int etf_N;
    bool st_normalize;
    bool ds_squared;
    QString filter_type;
    int n_a;
    int n_e;
    double sigma_dg;
    double sigma_rg;
    double sigma_dt;
    double sigma_rt;
    double bf_alpha;
    double precision_g;
    double precision_t;
    QString dog_type;
    QString dog_input;
    double sigma_e;
    double precision_e;
    double dog_k;
    double sigma_m;
    double precision_m;
    double step_m;
    double tau0;
    double tau1;
    double epsilon;
    double phi_e;
    QString dog_fgauss;
    bool dog_fgauss_adaptive;
    double dog_fgauss_max;
    int dog_N;
    QString dog_blend;
    int ag_N;
    double ag_k;
    double ag_p;

    bool quantization;
    QString quant_type;
    int nbins;
    double phi_q;
    double lambda_delta;
    double omega_delta;
    double lambda_phi;
    double omega_phi;
    bool warp_sharp;
    double sigma_w;
    double precision_w;
    double phi_w;
    bool final_smooth;
    QString final_type;
    double sigma_f;

    bool debug;
    bool draw_flow;
};
