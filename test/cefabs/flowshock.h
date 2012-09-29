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

class FlowShock : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Image and Video Abstraction by Coherence-Enhancing Filtering\n[Kyprianidis and Kang, 2011]")
    Q_CLASSINFO("description", "Eurographics 2011 paper research prototype")
    Q_CLASSINFO("category", "EG2011")
    Q_CLASSINFO("rating", "debug");
public:
    Q_INVOKABLE FlowShock();
    virtual QString caption() const;
    virtual void process();
    virtual void draw(ImageView *view, QPainter &p, int pass);
    virtual void dragBegin(ImageView *view, QMouseEvent* e);

    oz::gpu_image smoothedStructureTensor(const oz::gpu_image& img, /*const gpu_image& dog,*/ int n, int k);
    //gpu_image stFromCache(int f, int n, int k);

    void drawTFM(const oz::cpu_image& tfm, ImageView *view, QPainter &p, const QRect& R);
    void drawTFM2(const oz::cpu_image& st, ImageView *view, QPainter &p, const QRect& R);
    void drawTFMx(const oz::cpu_image& st, ImageView *view, QPainter &p, const QRect& R);
    void drawTFMA(const oz::cpu_image& tfm, ImageView *view, QPainter &p, const QRect& R, float pt);
    void drawTFME(const oz::cpu_image& st, ImageView *view, QPainter &p, const QRect& R, float pt);

    bool resample;
    int resample_w;
    int resample_h;
    bool auto_levels;
    double auto_levels_threshold;

    int total_N;
    bool remove_noise;
    QString noise_technique;
    int noise_radius;
    double noise_sigma_d;
    double noise_sigma_r;

    QString st_type;
    QString st_smoothing;
    double st_sigma_d;
    double st_sigma_r;
    double st_adaptive_threshold;
    int st_adaptive_N;
    bool st_normalize;
    bool st_flatten;
    double st_q;
    double st_relax;
    int st_mode;
    int moa_mode;
    int etf_N;
    bool ds_squared;

    bool fgauss;
    QString fgauss_type;
    bool fgauss_recalc_st;
    int fgauss_N;
    double fgauss_sigma_g;
    double fgauss_sigma_t;
    bool fgauss_adaptive;
    double fgauss_max;
    double fgauss_step;
    QString bf_type;
    double bf_sigma_d;
    double bf_sigma_r;

    bool shock;
    QString shock_type;
    bool shock_recalc_st;
    double blur_sigma_i;
    double blur_sigma_g;
    double shock_radius;
    QString dog_type;
    double dog_tau0;
    double dog_tau1;
    double dog_sigma_m;
    double dog_phi;
    int weickert_N;
    double weickert_step;

    bool smooth;
    QString smooth_type;
    double smooth_sigma;
    double blend;

    bool debug;
    bool draw_tf_fgauss;
    bool draw_tf_fgauss_relax;
    bool draw_tfx_fgauss;
    bool draw_tfA_fgauss;
    bool draw_tf_shock;
    bool draw_fgauss;
    bool draw_shock;
    bool draw_st;

    oz::gpu_image last_st;
    oz::gpu_image krnl41;
    oz::gpu_image krnl81;
    oz::gpu_image krnl84;
    oz::gpu_image noise;
    QPointF pos;
    QCache<unsigned, oz::gpu_image > st_cache;
};
