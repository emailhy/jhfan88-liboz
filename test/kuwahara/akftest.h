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

class AnisotropicKuwahara : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Anisotropic Kuwahara Filter\n[Kyprianidis et al., 2009]");
    Q_CLASSINFO("category", "Kuwahara Filter");
public:
    Q_INVOKABLE AnisotropicKuwahara();
    virtual void process();
    virtual void dragBegin(ImageView *view, QMouseEvent *e);
    virtual void draw(ImageView *view, QPainter &p, int pass);

public slots:
    void benchmark();
    void test();

public:
    int N;
    bool auto_levels;
    double auto_levels_threshold;
    bool noise;
    double variance;
    double noise_variance;
    oz::gpu_image noise_img;
    double rho;
    QString method;
    bool disable_st;
    double smoothing;
    double precision;
    int radius;
    double q;
    double alpha;
    double threshold;
    double a_star;
    bool debug;
    double scale_w;
    bool draw_st;
    bool draw_filter;
    bool show_weights;
    bool draw_isotropic;
    bool draw_origin;
    oz::gpu_image krnl41;
    oz::gpu_image krnl44;
    oz::gpu_image krnl81;
    oz::gpu_image krnl84;
    oz::gpu_image krnl8x2;
    double krnl_smoothing;
    double krnl_precision;
    int debug_x;
    int debug_y;
    ParamInt *pdebug_x;
    ParamInt *pdebug_y;
    int benchmark_n;
};


#if 0
class AnisotropicKuwahara2 : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Proposed Method");
    Q_CLASSINFO("category", "Kuwahara Filter");
public:
    Q_INVOKABLE AnisotropicKuwahara2();
    virtual void process();

    double rho;
    double zeta;
    double eta;
    int radius;
    double q;
    double alpha;
};


class AnisotropicKuwaharaDiff : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Difference to\n[Kyprianidis et al., 2009]");
    Q_CLASSINFO("category", "Kuwahara Filter");
public:
    Q_INVOKABLE AnisotropicKuwaharaDiff();
    virtual void process();

    double rho;
    double smoothing;
    double zeta;
    double eta;
    int radius;
    double q;
    double alpha;
    double diff;
    oz::gpu_image krnl;
    double krnl_smoothing;
    QImage jet_image;
};
#endif