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
#include "gkftest.h"
#include <oz/gkf_kernel.h>
#include <oz/gkf.h>
#include <oz/gkf2.h>
#include <oz/gkf_opt.h>
#include <oz/gkf_opt2.h>
#include <oz/polygkf.h>
#include <oz/colormap.h>
#include <oz/gauss.h>
#include <oz/minmax.h>
#include <oz/gpu_timer.h>
using namespace oz;


GeneralizedKuwahara::GeneralizedKuwahara() {
    krnl_smoothing = krnl_precision = 0;
    new ParamChoice(this, "method", "tex8", "tex4|poly4|tex8|tex8-ref|tex8-opt|tex8-opt2|poly8|", &method);
    new ParamInt   (this, "N", 1, 1, 100, 1, &N);
    new ParamDouble(this, "smoothing", 33.33f, 0, 100, 0.1, &smoothing);
    new ParamDouble(this, "precision", 2.5, 0, 5, 0.25f, &precision);
    new ParamInt   (this, "radius", 7, 1, 32, 1, &radius);
    new ParamDouble(this, "q", 8, 1, 16, 1, &q);
    new ParamDouble(this, "threshold", 1e-4, 0, 1, 1e-4, &threshold);
}


void GeneralizedKuwahara::process() {
    if ((krnl_smoothing != smoothing) || (krnl_precision != precision)) {
        krnl_smoothing = smoothing;
        krnl_precision = precision;

        //int half_width = (int)ceilf(precision * sigma_r);
        unsigned ksize = 32;
        float sigma_r = 0.5f * (ksize - 1) / precision;
        float sigma_s = sigma_r * smoothing / 100.0f;

        gpu_image krnl = gkf_char_function(0, 8, 0.5f * ksize, ksize);
        krnl.fill(0, 31, 0, 1, 32);
        publish("krnl0", colormap_jet(krnl));
        krnl = gauss_filter_xy(krnl, sigma_s, precision);
        publish("krnl1", colormap_jet(krnl));
        krnl = gkf_gaussian_mul(krnl, sigma_r, precision);
        krnl = krnl / max(krnl);
        krnl.fill(0, 31, 0, 1, 32);
        publish("krnl2", colormap_jet(krnl));
        krnl = circshift(krnl, 16, 16);
        publish("krnl3", colormap_jet(krnl));

        krnl41 = circshift(gkf_create_kernel1(32, smoothing / 100.0f, precision, 4), 16, 16);
        krnl81 = circshift(gkf_create_kernel1(32, smoothing / 100.0f, precision, 8), 16, 16);
        krnl84 = circshift(gkf_create_kernel4(32, smoothing / 100.0f, precision, 8), 16, 16);
        krnl8x2 = circshift(gkf_create_kernel8x2(32, smoothing / 100.0f, precision), 16, 16);
    }
    publish("krnl81", colormap_jet(krnl81 / max(krnl81)));

    gpu_image img = gpuInput0();
    publish("src", img);
    gpu_timer tt;
    for (int k = 0; k < N; ++k) {
        if (method == "tex4") {
            img = gkf_filter(img, krnl41, radius, q, threshold, 4);
        } else if (method == "tex8") {
            img = gkf_filter(img, krnl81, radius, q, threshold, 8);
        } else if (method == "tex8-ref") {
            img = gkf_filter2(img, krnl81, radius, q, threshold);
        } else if (method == "tex8-opt") {
            img = gkf_opt8_filter(img, krnl84, radius, q, threshold);
        } else if (method == "tex8-opt2") {
            img = gkf_opt8_filter2(img, krnl8x2, radius, q, threshold);
        } else if (method == "poly4") {
            img = polygkf(img, 4, radius, q, threshold, 2.0/radius, 0.84f);
        } else if (method == "poly8") {
            img = polygkf(img, 8, radius, q, threshold, 2.0/radius, 3.77f);
        }
    }
    double t = tt.elapsed_time();
    qDebug() << "GeneralizedKuwahara" << t << "ms";
    publish("$result", img);
}
