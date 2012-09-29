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
#include "anisodiff.h"
#include <oz/color.h>
#include <oz/colormap.h>
#include <oz/aniso_diff.h>
#include <oz/fgauss.h>
#include <oz/st.h>
#include <oz/dog.h>
#include <oz/ssia.h>
#include <oz/resample.h>
#include <oz/gpu_cache.h>
using namespace oz;


AnisoDiff::AnisoDiff() {
    new ParamInt   (this, "max_width", 512, 10, 4096, 16, &max_width);
    new ParamInt   (this, "max_height", 512, 10, 4096, 16, &max_height);
    new ParamDouble(this, "sigma", 0, 0, 10, 0.1, &sigma);
    new ParamChoice(this, "method", 0, "PM1|PM2", &method);
    new ParamDouble(this, "K", 0.015, 0, 1, 0.005, &K);
    new ParamDouble(this, "dt", 0.2, 0, 10, 0.1, &dt);
    new ParamInt   (this, "N", 10, 1, 1000, 1, &N);
}

void AnisoDiff::process() {
    gpu_image src = rgb2gray(resample_boxed(gpuInput0(), max_width, max_height));
    publish("src", src);
    gpu_image dst0 = aniso_diff(src, method, K, 0, N, dt);
    gpu_image dst1 = aniso_diff(src, method, K, sigma, N, dt);
    gpu_image diff = colormap_diff(dst0, dst1, 0.01f);
    publish("dst0", dst0);
    publish("dst1", dst1);
    publish("diff", diff);
    publish("$result", dst1);
}
