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
#include "ststat.h"
#include <oz/color.h>
#include <oz/noise.h>
#include <oz/hist.h>
#include <oz/shuffle.h>
#include <oz/st.h>
#include <oz/minmax.h>
using namespace oz;


StStat::StStat() {
    new ParamDouble(this, "rho", 2.0, 0.0, 10.0, 0.1, &rho);
    new ParamDouble(this, "H_min", -1.0, -10.0, 10.0, 0.1, &H_min);
    new ParamDouble(this, "H_max", 1.0, -10.0, 10.0, 0.1, &H_max);
    new ParamDouble(this, "H_max", 1.0, -10.0, 10.0, 0.1, &H_max);
    new ParamBool  (this, "normalize", false, &normalize);
}


void StStat::process() {
    gpu_image src = gpuInput0();
    gpu_image n = rgb2gray(gpuInput0());
    //gpu_image n = noise_uniform(512, 512, 0, 1);
    publish("n", n);
    std::vector<int> n_H = hist(n, 256, -1, 2);
    publishHistogram("n_H", n_H);

    gpu_image st = st_scharr_3x3(src, rho, normalize);
    gpu_image xx = shuffle(st, 0);

    float xx_min, xx_max;
    minmax(xx, &xx_min, &xx_max);
    qDebug() << "--min:" << xx_min << "max:" << xx_max;

    std::vector<int> xx_H = hist(xx, 256, -1, 1);
    publishHistogram("xx_H", xx_H);

    gpu_image sqxx = sqrt(xx);
    minmax(sqxx, &xx_min, &xx_max);
    qDebug() << "  min:" << xx_min << "max:" << xx_max;

    std::vector<int> sqxx_H = hist(sqxx, 256, -0.25, 0.25);
    publishHistogram("sqxx_H", sqxx_H);
}
