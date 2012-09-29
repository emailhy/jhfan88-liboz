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
#include "bilateral.h"
#include <oz/color.h>
#include <oz/bilateral.h>
#include <oz/oabf.h>
#include <oz/gauss.h>
#include <oz/shuffle.h>
#include <oz/make.h>
#include <oz/resample.h>
#include <oz/gpu_cache.h>
#include <oz/st.h>
using namespace oz;


BilateralFilter::BilateralFilter() {
    new ParamInt   (this, "max_width", 512, 10, 4096, 16, &max_width);
    new ParamInt   (this, "max_height", 512, 10, 4096, 16, &max_height);
    new ParamInt   (this, "N", 4, 0, 50, 1, &N);
    new ParamChoice(this, "colorspace", "CIELAB", "CIELAB|RGB", &colorspace);
    new ParamBool  (this, "per_channel", false, &per_channel);
    new ParamDouble(this, "sigma_d", 3, 0, 20, 1, &sigma_d);
    new ParamDouble(this, "sigma_r", 4.25, 0, 100, 1, &sigma_r);
    new ParamDouble(this, "rgb_scale", 0.01, 0, 1, 0.005, &rgb_scale);
    new ParamDouble(this, "precision", 3, 1, 10, 0.25, &precision);
}


void BilateralFilter::process() {
    gpu_image src = gpuInput0();
    if (((int)src.w() > max_width) || ((int)src.h() > max_height)) {
        double zw = 1.0 * qMin((int)src.w(), max_width) / src.w();
        double zh = 1.0 * qMin((int)src.h(), max_height) / src.h();
        int w, h;
        if (zw <= zh) {
            w = max_width;
            h = (int)(zw * src.h());
        } else {
            w = (int)(zh * src.w());
            h = max_height;
        }
        src = resample(src, w, h, RESAMPLE_LANCZOS3);
        gpu_cache_clear();
    }
    publish("src", src);
    gpu_image img = (colorspace == "CIELAB")? rgb2lab(src) : src;
    for (int i = 0; i < N; ++i) {
        if (!per_channel) {
            img = bilateral_filter(img, sigma_d, (colorspace == "CIELAB")? sigma_r : sigma_r * rgb_scale, precision);
        } else {
            gpu_image c[3];
            for (int i = 0; i < 3; ++i) {
                c[i] = shuffle(img, i);
                c[i] = bilateral_filter(c[i], sigma_d, (colorspace == "CIELAB")? sigma_r : sigma_r * rgb_scale, precision);
            }
            img = make(c[0], c[1], c[2]);
        }
    }
    publish("$result", (colorspace == "CIELAB")? lab2rgb(img) : img);
}
