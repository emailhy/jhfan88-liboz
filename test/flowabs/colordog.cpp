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
#include "colordog.h"
#include <oz/color.h>
#include <oz/shuffle.h>
#include <oz/make.h>
#include <oz/minmax.h>
#include <oz/st.h>
#include <oz/stgauss2.h>
#include <oz/stbf2.h>
#include <oz/dog.h>
#include <oz/highpass.h>
#include <oz/oabf.h>
#include <oz/bilateral.h>
#include <oz/ivacef.h>
#include <oz/grad.h>
#include <oz/blend.h>
#include <oz/beltrami.h>
#include <oz/gauss.h>
#include <oz/noise.h>
#include <oz/stgauss2.h>
#include <oz/hist.h>
using namespace oz;


/*
gpu_image<float> mag_diff( const gpu_image<float>& src0, const gpu_image<float>& src1 );
*/
gpu_image color_gdog( const gpu_image& src, const gpu_image& tfab,
                      float sigma_e, float sigma_r, float precision, float tau );

gpu_image chroma_sharp( const gpu_image& L, const gpu_image& hp,
                         float K, float B1, float B2 );



gpu_image stflow( const gpu_image& st ) {
    gpu_image noise = noise_normal(st.w(), st.h());
    gpu_image flow = stgauss2_filter(noise, st, 6, 22.5f, false, true, true, 2, 0.25f);
    flow = hist_eq(flow);
    flow = stgauss2_filter(flow, st, 1.5f, 22.5f, false, true, true, 2, 0.25f);
    return flow;
}


ColorDog::ColorDog() {
    ParamGroup *g;
    g = new ParamGroup(this, "structure tensor");
    new ParamChoice(g, "st_type",      "scharr", "scharr|relaxed", &st_type);
    new ParamDouble(g, "rho", 1.0, 0.0, 10.0, 0.05, &rho);
    new ParamDouble(g, "tau_r", 0.002, 0.0, 1.0, 0.001, &tau_r);

    g = new ParamGroup(this, "pre-process");
    new ParamInt   (g, "N", 4, 0, 100, 1, &N);
    new ParamChoice(g, "pre-smooth", "bf", "none|bf|oabf|fbl|beltrami|stgauss", &pre_smooth);
    new ParamDouble(g, "sigma_g", 3.0, 0.0, 20.0, 1, &sigma_g);
    new ParamDouble(g, "sigma_t", 6.0, 0.0, 20.0, 1, &sigma_t);
    new ParamDouble(g, "sigma_r", 4.25f, 0.0, 100.0, 1, &sigma_r);
    new ParamDouble(g, "max_angle", 22.5, 0.0, 90.0, 1, &max_angle);
    new ParamDouble(g, "precision", 2.0, 1.0, 10.0, 1, &precision);

    g = new ParamGroup(this, "dog");
    new ParamChoice(g, "dog_type",  "color", "mono|color", &dog_type);
    new ParamChoice(g, "luminance", "nvac", "L|gray|nvac", &luminance);
    new ParamDouble(g, "sigma_e", 1, 0, 20, 0.005, &sigma_e);
    new ParamDouble(g, "precision_e", 3, 1, 5, 0.1, &precision_e);
    new ParamDouble(g, "k", 1.6, 0, 100, 0.05, &k);
    new ParamDouble(g, "sigma_m", 3, 0, 20, 1, &sigma_m);
    new ParamDouble(g, "tau", 0.98, 0, 1, 0.005, &tau);
    new ParamDouble(g, "epsilon", 0, -10, 10, 0.005, &epsilon);
    new ParamDouble(g, "phi_e",   2, 0, 1000, 0.1, &phi_e);
    new ParamDouble(g, "sigma_f", 1.0, 0, 10, 1, &sigma_f);
}


void ColorDog::process() {
    gpu_image src = gpuInput0();
    publish("src", src);

    gpu_image st;
    if (st_type == "scharr") {
        st = st_scharr_3x3( src, rho );
    } else {
        st = ivacef_compute_st( src, gpu_image(), rho, tau_r, 1);
    }
    gpu_image lfm = st_lfm(st);
    publish("flow", stflow(st));

    gpu_image lab = rgb2lab(src);
    for (int i = 0; i < N; ++i) {
        if (pre_smooth == "bf") {
            lab = bilateral_filter(lab, sigma_g, sigma_r, precision);
        } else if (pre_smooth == "oabf") {
            lab = oabf_1d(lab, lfm, sigma_g, sigma_r, false, true, true, precision);
            lab = oabf_1d(lab, lfm, sigma_t, sigma_r, true, true, true, precision);
        } else if (pre_smooth == "fbl") {
            lab = oabf_1d(lab, lfm, sigma_g, sigma_r, false, true, true, 3);
            lab = stbf2_filter(lab, st, sigma_t, sigma_r, precision, max_angle, false, true, true, 2, 1.0f);
        } else if (pre_smooth == "beltrami") {
            lab = beltrami(lab, 10);
        } else if (pre_smooth == "stgauss") {
            lab = stgauss2_filter(lab, st, sigma_t, max_angle, false, true, true, 2, 1.0f);
        }
    }
    publish("lab-smooth", lab2rgb(lab));

    gpu_image n = shuffle(lab, 0);
    gpu_image a = shuffle(lab, 1);
    gpu_image b = shuffle(lab, 2);
    if (luminance == "nvac") {
        n = rgb2nvac(lab2rgb(lab));
    } else if (luminance == "gray") {
        n = rgb2gray(lab2rgb(lab));
    }
    publish("n", n / 100.0f);

    gpu_image nab = make(n, a, b);

    if (dog_type == "color") {
        gpu_image cdog = color_gdog(nab, lfm, sigma_e, sigma_e*k, precision_e, tau);
        cdog = stgauss2_filter(cdog, st, sigma_m, 90.0f, false, true, true, 2, 1.0f);
        publish("cdog-sign", dog_colorize(shuffle(cdog,0)));

        gpu_image Ls = chroma_sharp(n, cdog, 1, 15, 40);
        publish("n-sign", dog_colorize(Ls));

        Ls = Ls + gauss_filter_xy(n, 1.6f) * (1.0f - tau);
        Ls = stgauss2_filter(Ls, st, sigma_m, 90.0f, false, true, true, 2, 1.0f);

        Ls = dog_threshold_tanh(Ls, epsilon, phi_e);
        Ls = stgauss2_filter(Ls, st, sigma_f, 90.0f, false, true, true, 2, 1.0f);
        publish("$result", Ls);
    } else {
        gpu_image dog = gradient_dog(shuffle(nab, 0), st_to_tangent(st), sigma_e, sigma_e*k, tau, 0.0f, precision_e);
        dog = stgauss2_filter(dog, st, sigma_m, 90.0f, false, true, true, 2, 1.0f);
        publish("n-sign", dog_colorize(dog));

        dog = dog_threshold_tanh(dog, epsilon, phi_e);
        dog = stgauss2_filter(dog, st, sigma_f, 90.0f, false, true, true, 2, 1.0f);
        publish("$result", dog);
    }

}


