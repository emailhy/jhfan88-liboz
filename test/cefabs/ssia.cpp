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
#include "ssia.h"
#include <oz/color.h>
#include <oz/gauss.h>
#include <oz/fgauss.h>
#include <oz/st.h>
#include <oz/dog.h>
#include <oz/ssia.h>
#include <oz/cmcf.h>
#include <oz/etf2.h>
#include <oz/dog_shock.h>
#include <oz/colormap.h>
using namespace oz;


SSIA::SSIA() {
    new ParamDouble(this, "pre_smooth", 0, 0, 10, 0.1, &pre_smooth);
    new ParamInt   (this, "total_N", 5, 1, 100, 1, &total_N);
    new ParamInt   (this, "etf_halfw", 3, 0, 10, 1, &etf_halfw);
    new ParamInt   (this, "etf_N", 2, 0, 10, 1, &etf_N);
    new ParamInt   (this, "cmcf_N", 10, 1, 1000, 1, &cmcf_N);
    new ParamDouble(this, "cmcf_weight", 1, 0,1, 0.1, &cmcf_weight);
    new ParamDouble(this, "shock_sigma", 1, 0, 10, 0.1, &shock_sigma);
    new ParamDouble(this, "shock_tau", 1, 0, 2, 0.1, &shock_tau);
    new ParamDouble(this, "final_smooth", 0, 0, 2, 0.1, &final_smooth);
}


void SSIA::process() {
    gpu_image src = gpuInput0();
    publish("src", src);

    src = gauss_filter_xy(src, pre_smooth);
    publish("src-smooth", src);

    gpu_image dst = ssia(
        src,
        total_N,
        cmcf_N,
        cmcf_weight,
        etf_N,
        etf_halfw,
        shock_sigma,
        shock_tau
    );
    if (final_smooth > 0) {
        gpu_image tm = st_to_tangent(st_scharr_3x3(dst, 2));
        dst = fgauss_filter(dst, tm, final_smooth);
    }
    publish("$result", dst);
}


SSIA2::SSIA2() {
    ParamGroup *g;
    new ParamDouble(this, "pre_smooth", 0, 0, 10, 0.1, &pre_smooth);
    new ParamInt   (this, "total_N", 10, 1, 100, 1, &total_N);

    g = new ParamGroup(this, "flow_field");
    new ParamChoice(g, "flow_type", "st", "etf|st" , &flow_type);
    new ParamDouble(g, "st_sigma", 2, 0, 10, 0.5, &st_sigma);
    new ParamInt   (g, "etf_halfw", 3, 0, 10, 1, &etf_halfw);
    new ParamInt   (g, "etf_N", 2, 0, 10, 1, &etf_N);

    g = new ParamGroup(this, "mcf");
    new ParamChoice(g, "cmcf_type", "original", "original|improved" , &cmcf_type);
    new ParamInt   (g, "cmcf_N", 5, 1, 100, 1, &cmcf_N);
    new ParamDouble(g, "cmcf_step", 1, 0,1, 0.1, &cmcf_step);
    new ParamDouble(g, "cmcf_weight", 1, 0,1, 0.1, &cmcf_weight);
    new ParamDouble(g, "shock_sigma", 1, 0, 10, 0.1, &shock_sigma);
    new ParamDouble(g, "shock_tau", 1, 0, 2, 0.1, &shock_tau);

    new ParamDouble(this, "final_smooth", 0, 0, 2, 0.1, &final_smooth);
}


void SSIA2::process() {
    gpu_image img = gpuInput0();
    publish("$source", img);

    gpu_image gray;
    gpu_image dog;
    for (int l = 0; l < total_N; ++l) {
        gpu_image etf;
        if (flow_type == "st") {
            etf = st_to_tangent(st_scharr_3x3(img, st_sigma));
        } else {
            gray = rgb2gray(img);
            etf = etf_xy2(gray, etf_halfw, etf_N);
        }
        for (int k = 0; k < cmcf_N; ++k) {
            if (cmcf_type == "improved") {
                img = cmcf(img, etf, cmcf_step, cmcf_weight);
            } else {
                img = ssia_cmcf(img, etf, cmcf_weight);
            }
        }

        if (shock_sigma > 0) {
            gray = rgb2gray(img);
            dog = dog_filter(gray, shock_sigma, shock_sigma * 1.6f, shock_tau, 0);
            img = ssia_shock(img, dog);
        }
    }
    if (final_smooth > 0) {
        gpu_image tm = st_to_tangent(st_scharr_3x3(img, 2));
        img = fgauss_filter(img, tm, final_smooth);
    }
    publish("$result", img);
}
