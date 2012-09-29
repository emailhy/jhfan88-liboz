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
#include "wog.h"
#include <oz/io.h>
#include <oz/color.h>
#include <oz/gauss.h>
#include <oz/bilateral.h>
#include <oz/shuffle.h>
#include <oz/wog.h>
#include <oz/blend.h>
#include <oz/grad.h>
using namespace oz;


WOG::WOG() {
    ParamGroup *g;

    g = new ParamGroup(this, "bilateral_filter");
    new ParamChoice(g, "filter_type", "xy", "xy|full", &filter_type);
    new ParamInt   (g, "n_e",     1, 0, 20, 1, &n_e);
    new ParamInt   (g, "n_a",     4, 0, 20, 1, &n_a);
    new ParamDouble(g, "sigma_d", 3, 0, 20, 1, &sigma_d);
    new ParamDouble(g, "sigma_r", 4.25, 0, 100, 1, &sigma_r);

    g = new ParamGroup(this, "dog", true, &dog);
    new ParamDouble(g, "sigma_e", 1, 0, 20, 0.005, &sigma_e);
    new ParamDouble(g, "precision_e", 2, 1, 5, 0.1, &precision_e);
    new ParamDouble(g, "tau",     0.99, 0, 2, 0.005, &tau);
    new ParamDouble(g, "phi_e",   2, 0, 20, 0.1, &phi_e);
    new ParamDouble(g, "epsilon", 0, -10, 10, 0.005, &epsilon);
    new ParamChoice(g, "background", "color", "none|color|source", &background);

    g = new ParamGroup(this, "quantization", true, &quantization);
    new ParamChoice(g, "quant_type", "adaptive", "fixed|adaptive", &quant_type);
    new ParamInt   (g, "nbins", 8, 1, 255, 1, &nbins);
    new ParamDouble(g, "phi_q", 2, 0, 100, 0.025, &phi_q);
    new ParamDouble(g, "lambda_delta", 0, 0, 100, 1, &lambda_delta);
    new ParamDouble(g, "omega_delta", 2, 0, 100, 1, &omega_delta);
    new ParamDouble(g, "lambda_phi", 0.9, 0, 100, 1, &lambda_phi);
    new ParamDouble(g, "omega_phi", 1.6, 0, 100, 1, &omega_phi);

    g = new ParamGroup(this, "warp_sharp", true, &warp_sharp);
    new ParamDouble(g, "sigma_w", 1.5, 0, 20, 1, &sigma_w);
    new ParamDouble(g, "precision_w", 2, 1, 5, 0.1, &precision_w);
    new ParamDouble(g, "phi_w", 2.7, 0, 100, 0.025, &phi_w);
}


void WOG::process() {
    gpu_image src = gpuInput0();
    publish("src", src);

    gpu_image X = shuffle(src, 0);
    gpu_image Y = X;
    publish("X", X);
    publish("Y", Y);

    gpu_image img = rgb2lab(src);
    gpu_image Ie = img;
    gpu_image Ia = img;

    int N = std::max(n_e, n_a);
    for (int i = 0; i < N; ++i) {
        if (filter_type == "full") {
            img = bilateral_filter(img, sigma_d, sigma_r);
        } else {
            img = bilateral_filter_xy(img, sigma_d, sigma_r);
        }
        if (i == (n_e - 1)) Ie = img;
        if (i == (n_a - 1)) Ia = img;
    }

    publish("Ie", lab2rgb(Ie));
    publish("Ia", lab2rgb(Ia));

    gpu_image L;
    if (dog) {
        L = shuffle(Ie, 0);
        publish("L0", L / 100);
        L = wog_dog( L, sigma_e, 1.6*sigma_e, tau, phi_e, epsilon, precision_e );
        publish("dog", L);
    }

    if (quantization) {
        if (quant_type == "fixed") {
            Ia = wog_luminance_quant( Ia, nbins, phi_q );
        } else {
            Ia = wog_luminance_quant( Ia, nbins, lambda_delta, omega_delta, lambda_phi, omega_phi );
        }
        publish("quantization", lab2rgb(Ia));
    }

    img = lab2rgb(Ia);

    if (dog) {
        if (background == "color")
            img = blend_intensity(img, L, BLEND_MULTIPLY);
        else if (background == "source")
            img = blend_intensity(src, L, BLEND_MULTIPLY);
        else
            img = gray2rgb(L);
    }

    if (warp_sharp) {
        gpu_image S = grad_sobel_mag(img);
        S = gauss_filter(S, sigma_w, precision_w);
        publish("S", S);
        img = wog_warp(img, S, phi_w);
    }

    publish("$result", img);
}
