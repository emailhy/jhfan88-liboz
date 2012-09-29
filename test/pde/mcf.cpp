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
#include "mcf.h"
#include <oz/color.h>
#include <oz/colormap.h>
#include <oz/gauss.h>
#include <oz/fgauss.h>
#include <oz/st.h>
#include <oz/dog.h>
#include <oz/ssia.h>
#include <oz/mcf.h>
#include <oz/gmcf.h>
#include <oz/curvature.h>
#include <oz/test_pattern.h>
#include <oz/minmax.h>
#include <oz/hist.h>
using namespace oz;


MCF::MCF() {
    new ParamInt   (this, "N", 5, 0, 10000, 10, &N);
    new ParamDouble(this, "sigma", 0, 0, 10, 0.1, &sigma);
    new ParamDouble(this, "step", 0.25, 0, 2, 0.01, &step);
    new ParamChoice(this, "type", "mcf", "mcf|gmcf_sp|gmcf_pm", &type);
    new ParamDouble(this, "epsilon", 0, 0, 2, 1e-5, &epsilon);
    new ParamDouble(this, "lambda", 1, 0, 10, 0.01, &lambda);
    new ParamDouble(this, "p", 1, 0, 10, 0.01, &p);
}


void MCF::process() {
    gpu_image src = rgb2gray(gpuInput0());
    publish("src", src);

    gpu_image img = gauss_filter_xy(src, sigma, 10);
    publish("$input", img);

    for (int k = 0; k < N; ++k) {
        if (type == "mcf") {
            img = mcf(img, step, epsilon);
        }
        else if (type == "gmcf_sp") {
            img = gmcf_sp(img, p, step);
        }
        else if (type == "gmcf_pm") {
            img = gmcf_pm(img, lambda, step);
        }
    }

    publish(QString("$result"), img);
}



/*
void MCF::process() {
    gpu_image src = gpuInput0();
    gpu_image gray = rgb2gray(src);
    publish("gray", gray);

    gpu_image img = gray;
    //img = test_line(512,512, angle, 5);
    //img = test_wiggle(512, 512, 100);
    //img = test_circle(512, 512, 100);
    img = gauss_filter_xy(img, sigma, 10);
    publish("$input", img);

    for (int k = 0; k < N; ++k) {
        //img = mcf(img, step, epsilon);
        //img = gmcf(img, lambda, step);
    }

    //gpu_image kappa = curvature(img, epsilon);
    //std::vector<int> H = histogram(kappa, 256, -1, 2);
    //publishHistogram("kappa-H", H);
    //publish(QString("$kappa"), dog_colorize(kappa));
    //publish(QString("$kappaC"), colormap_jet( kappa/200.0 + 0.5f));
    publish(QString("$mcf"), img);
}
*/