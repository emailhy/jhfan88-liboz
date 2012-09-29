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
#include <vector>
#include <algorithm>
#include <oz/st.h>

class MsAkfTest : public SimpleModule {
    Q_OBJECT
    Q_CLASSINFO("title", "Image and Video Abstraction by Multi-scale Anisotropic Kuwahara Filtering\n[Kyprianidis, 2011]")
    Q_CLASSINFO("category", "Kuwahara Filter");
    Q_CLASSINFO("rating", "debug");
public:
    Q_INVOKABLE MsAkfTest();
    virtual void process();
    //virtual void dragBegin(ImageView *view, QMouseEvent *e);
    virtual void draw(ImageView *view, QPainter &p, int pass);

    bool auto_levels;
    double auto_levels_threshold;
    bool noise;
    double variance;
    oz::gpu_image krnl4;
    QString pyr_factor;
    int pyr_down;
    int pyr_up;
    bool shock;
    double scale;
    double damping;
    double vthresh;
    double threshold;
    int prop_mode;
    double rho;
    int radius;
    double q;
    double alpha;
    bool st_enable_ms;
    double st_epsilon;
    oz::moa_t moa;
    bool st_indep_prop;
    bool debug;
    int orientation;
    int showLevel;
    bool draw_scale;

    void publishPyr(const QString& key, const std::vector<oz::gpu_image>& P);
};
