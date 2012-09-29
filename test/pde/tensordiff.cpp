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
#include "tensordiff.h"
#include <oz/color.h>
#include <oz/tensor_diff.h>
#include <oz/minmax.h>
#include <oz/resample.h>
#include <oz/gpu_cache.h>
using namespace oz;


EdgeEnhancingDiff::EdgeEnhancingDiff() {
    new ParamInt   (this, "max_width", 512, 10, 4096, 16, &max_width);
    new ParamInt   (this, "max_height", 512, 10, 4096, 16, &max_height);
    new ParamBool  (this, "grayscale", false, &grayscale);
    new ParamDouble(this, "sigma", 3, 0, 10, 0.1, &sigma);
    new ParamDouble(this, "rho", 0, 0, 10, 0.1, &rho);
    new ParamDouble(this, "lambda", 3.6, 0, 10, 0.01, &lambda);
    new ParamDouble(this, "dt", 0.2, 0, 10, 0.1, &dt);
    new ParamInt   (this, "N", 1, 1, 1000, 1, &N);
}

void EdgeEnhancingDiff::process() {
    gpu_image src = resample_boxed(gpuInput0(), max_width, max_height);
    if (grayscale) src = rgb2gray(src);
    publish("src", src);
    gpu_image dst = ee_diff(src, sigma, rho, lambda/255.0f, dt, N);
    publish("$result", dst);
    publish("result-normalized", normalize(dst));
}


CoherenceEnhancingDiff::CoherenceEnhancingDiff() {
    new ParamInt   (this, "max_width", 512, 10, 4096, 16, &max_width);
    new ParamInt   (this, "max_height", 512, 10, 4096, 16, &max_height);
    new ParamBool  (this, "grayscale", false, &grayscale);
    new ParamDouble(this, "sigma", 3, 0, 10, 0.1, &sigma);
    new ParamDouble(this, "rho", 0, 0, 10, 0.1, &rho);
    new ParamDouble(this, "alpha", 0.001, 0, 10, 0.001, &alpha);
    new ParamDouble(this, "C", 1, 0, 10, 0.001, &C);
    new ParamInt   (this, "m", 1, 0, 10, 1, &m);
    new ParamDouble(this, "dt", 0.2, 0, 10, 0.1, &dt);
    new ParamInt   (this, "N", 1, 1, 1000, 1, &N);
}

void CoherenceEnhancingDiff::process() {
    gpu_image src = resample_boxed(gpuInput0(), max_width, max_height);
    if (grayscale) src = rgb2gray(src);
    publish("$src", src);
    gpu_image dst = ce_diff(src, sigma, rho, alpha/255.0f, C/255.0f/255.0f, m, dt, N);
    publish("$result", dst);
    publish("result-normalized", normalize(dst));
}
