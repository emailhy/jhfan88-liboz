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
#include "shock.h"
#include <oz/color.h>
#include <oz/colormap.h>
#include <oz/gauss.h>
#include <oz/aniso_gauss.h>
#include <oz/fgauss.h>
#include <oz/st.h>
#include <oz/dog.h>
#include <oz/ssia.h>
#include <oz/dog_shock.h>
#include <oz/shock_flt.h>
#include <oz/deriv2nd.h>
#include <oz/noise.h>
#include <oz/io.h>
using namespace oz;


ShockFDoGUpwind::ShockFDoGUpwind() {
    new ParamInt   (this, "N", 10, 0, 100, 1, &N);
    new ParamDouble(this, "rho", 5.0, 0.0, 10.0, 0.5, &rho);
    new ParamDouble(this, "sigma", 4.0, 0.0, 10.0, 0.5, &sigma);
    new ParamDouble(this, "fdog_sigma", 0.9, 0.0, 10.0, 0.25, &fdog_sigma);
    new ParamDouble(this, "fdog_tau", 1.0, 0.0, 2.0, 0.001, &fdog_tau);
    new ParamDouble(this, "step", 0.4, 0, 10, 0.1, &step);
}


void ShockFDoGUpwind::process() {
    gpu_image img = gpuInput0();
    gpu_image dog;
    for (int k = 0; k < N; ++k) {
        gpu_image gray = rgb2gray(img);
        gpu_image tm = st_to_tangent(st_scharr_3x3(gray, rho));
        gray = gauss_filter_xy(gray, sigma);
        dog = gradient_dog(gray, tm, fdog_sigma, fdog_sigma*1.6f, fdog_tau, 0);
        img = dog_shock_upwind(img, dog, step);
    }
    publish("sign", dog_sign(dog));
    publish("$result", img);
}


WeickertCohEnhShock::WeickertCohEnhShock() {
    new ParamInt   (this, "N", 10, 0, 100, 1, &N);
    new ParamDouble(this, "rho", 3.0, 0.0, 10.0, 0.5, &rho);
    new ParamChoice(this, "smoothing", "isotropic", "isotropic|anisotropic", &smooting);
    new ParamDouble(this, "alpha", 1.0, 0, 1000, 1, &alpha);
    new ParamDouble(this, "sigma", 2.0, 0.0, 10.0, 0.5, &sigma);
    new ParamDouble(this, "step", 0.4, 0, 10, 0.1, &step);
}


void WeickertCohEnhShock::process() {
    gpu_image img = gpuInput0();
    gpu_image sign;
    for (int k = 0; k < N; ++k) {
        gpu_image st = st_scharr_3x3(img, rho);

        gpu_image blur;
        if (smooting == "isotropic")
            blur = gauss_filter_xy(img, sigma);
        else  {
            gpu_image lfm = st_lfm(st, alpha);
            blur = aniso_gauss(img, lfm, sigma, 0, 2);
        }
        deriv2nd_t d = deriv2nd(blur);

        gpu_image tm = st_to_gradient(st);
        sign = deriv2nd_sign(tm, d.Ixx, d.Ixy, d.Iyy);
        img = dog_shock_upwind(img, sign, step);
    }
    publish("sign", sign);
    publish("$result", img);
}


OsherRudinShock::OsherRudinShock() {
    new ParamInt   (this, "N", 10, 0, 1000, 1, &N);
    new ParamDouble(this, "dt", 0.2, 0.0, 10.0, 0.01, &dt);
}


void OsherRudinShock::process() {
    gpu_image src = rgb2gray(gpuInput0());
    publish("src", src);
    gpu_image dst = shock_flt_or(src, N, dt);
    publish("$result", dst);
}


AlvarezMazorraShock::AlvarezMazorraShock() {
    new ParamBool  (this, "pre_blur", true, &pre_blur);
    new ParamInt   (this, "N", 50, 0, 1000, 1, &N);
    new ParamDouble(this, "c", 0.2, 0.0, 10.0, 0.01, &c);
    new ParamDouble(this, "sigma", sqrt(10.0), 0.0, 20.0, 0.01, &sigma);
    new ParamDouble(this, "dt", 0.1, 0.0, 10.0, 0.01, &dt);
}


void AlvarezMazorraShock::process() {
    gpu_image src = rgb2gray(gpuInput0());
    publish("src", src);
    publish("src_smooth", gauss_filter_xy(src, sigma));
    gpu_image dst = shock_flt_am(src, c, sigma, N, dt);
    publish("$result", dst);
}


