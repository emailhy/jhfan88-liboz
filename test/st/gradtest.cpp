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
#include "gradtest.h"
#include <oz/color.h>
#include <oz/colormap.h>
#include <oz/shuffle.h>
#include <oz/grad.h>
#include <oz/noise.h>
#include <oz/test_pattern.h>
#include <oz/qpainter_draw.h>
using namespace oz;


GradTest::GradTest() {
    noise_var = 0;
    new ParamChoice(this, "input", "sphere", "sphere|sin|cos|image", &input);
    new ParamChoice(this, "filter", "central", "central|sobel|scharr-3x3|scharr-5x5|gaussian", &filter);
    new ParamDouble(this, "sigma", 2.0, 0.0, 10.0, 0.1, &sigma);
    new ParamDouble(this, "precision", 3.0, 1.0, 10.0, 0.1, &precision);
    new ParamDouble(this, "scale", 1.0, 0.0, 100.0, 0.1, &scale);
    new ParamDouble(this, "variance",  0, 0, 1, 0.0005, &variance);
    new ParamBool  (this, "draw_field", false, &draw_field);
    new ParamBool  (this, "draw_orientation", false, &draw_orientation);
}


void GradTest::process() {
    {
        gpu_image I = test_grad3(128,128);
        gpu_image B = shuffle(I, 0);
        gpu_image G = shuffle(I, 1);
        gpu_image R = shuffle(I, 2);
        publish("3I", I);
        publish("3_R", R);
        publish("3_G", G);
        publish("3_B", B);
    }

    gpu_image src =
        (input == "sphere")? test_sphere(512, 512) :
        (input == "sin")? test_simple(512, 512, 0, 10, 1, 1) :
        (input == "cos")? test_simple(512, 512, 0, 10, 1, 0) :
        rgb2gray(gpuInput0());

    if (variance > 0) {
        if (!noise.is_valid() || (noise.size() != src.size()) || (noise_var != variance)) {
            noise = noise_normal(src.w(), src.h(), 0, variance);
            noise_var = variance;
        }
        publish("noise", noise);
        src = src + noise;
    }
    publish("src", src);

    gpu_image g;
    if (filter == "central") {
        g = grad_central_diff(src, false);
    } else if (filter == "sobel") {
        g = grad_sobel(src, false);
    } else if (filter == "scharr-3x3") {
        g = grad_scharr_3x3(src, false);
    } else if (filter == "scharr-5x5") {
        g = grad_scharr_5x5(src, false);
    } if (filter == "gaussian") {
        g = grad_gaussian(src, sigma, precision, false);
    }
    gpu_image gx = shuffle(g, 0);
    gpu_image gy = shuffle(g, 1);
    publish("g", g / abs(g));

    publish("gx", colormap_jet(0.5f/scale*gx+0.5f));
    publish("gy", colormap_jet(0.5f/scale*gy+0.5f));

    gpu_image a = grad_angle(g) / CUDART_PI_F;
    publish("a", 0.5f*a+0.5f);

    {
        cpu_image cs = src.cpu();
        int W = cs.w();
        QImage I(W, 200, QImage::Format_RGB32);
        QPainter p(&I);
        p.drawLine(0,100, W, 100);
        p.drawLine(W/2,0, W/2, 200);

        int y = cs.h() / 2;
        {
            QPainterPath path;
            p.setPen(Qt::red);
            for (int x = 0; x < W; ++x) {
                int f = 100 - (cs.at<float>(x,y)-0.5f) * 100;
                if (x == 0) path.moveTo(x, f); else path.lineTo(x, f);
            }
            p.drawPath(path);
        }
        {
            QPainterPath path;
            p.setPen(Qt::blue);
            cpu_image cgx = gx.cpu();
            for (int x = 0; x < W; ++x) {
                int f = 100 - cgx.at<float>(x,y) * 300;
                if (x == 0) path.moveTo(x, f); else path.lineTo(x, f);
            }
            p.drawPath(path);
        }

        publish("plot", I);
    }
}


void GradTest::draw(ImageView *view, QPainter &p, int pass) {
    Module::draw(view, p, pass);

    QRectF R = p.clipBoundingRect();
    QRect aR = R.toAlignedRect().intersected(view->image().rect());

    double px = draw_pt2px(p);
    if (draw_field && (view->zoom() > 3) && pass) {
        p.save();
        p.setPen(QPen(Qt::red, 0.25*px));
        cpu_image g = publishedImage("g");
        draw_vector_field(p, g, aR, true);
        p.restore();
    }
    if (draw_orientation && (view->zoom() > 3) && pass) {
        p.save();
        p.setPen(QPen(Qt::blue, 0.25*px));
        cpu_image g = publishedImage("g");
        draw_orientation_field(p, g, aR);
        p.restore();
    }
}