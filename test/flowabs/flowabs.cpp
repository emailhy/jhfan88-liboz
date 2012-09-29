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
#include "flowabs.h"
#include <oz/color.h>
#include <oz/gauss.h>
#include <oz/shuffle.h>
#include <oz/resample.h>
#include <oz/st.h>
#include <oz/bilateral.h>
#include <oz/fgauss.h>
#include <oz/oabf.h>
#include <oz/dog.h>
#include <oz/wog.h>
#include <oz/blend.h>
#include <oz/noise.h>
#include <oz/hist.h>
#include <oz/gpu_timer.h>
#include <oz/gpu_cache.h>
#include <oz/ds.h>
#include <oz/etf.h>
#include <oz/stgauss2.h>
#include <oz/stbf2.h>
#include <oz/noise.h>
#include <oz/apparent_gray.h>
#include <oz/qpainter_draw.h>
using namespace oz;


FlowAbs::FlowAbs() {
    ParamGroup *g;

    new ParamChoice(this, "output", "fill+edges", "edges|fill|fill+edges", &output);
    new ParamChoice(this, "input_gamma", "srgb", "srgb|linear-rgb", &input_gamma);
    new ParamInt   (this, "max_width", 512, 10, 4096, 16, &max_width);
    new ParamInt   (this, "max_height", 512, 10, 4096, 16, &max_height);
    g = new ParamGroup(this, "auto_levels", false, &auto_levels);
    new ParamDouble(g, "threshold", 0.1, 0, 100, 0.05, &auto_levels_threshold);

    g = new ParamGroup(this, "noise",  false, &noise);
    new ParamDouble(g, "variance",     0.01, 0, 1, 0.005, &variance);

    g = new ParamGroup(this, "structure_tensor");
    new ParamChoice(g, "st_type",      "scharr-rgb", "central-diff|sobel-rgb|sobel-lab|sobel-L|scharr-rgb|gaussian-deriv|etf-full|etf-xy", &st_type);
    new ParamDouble(g, "rho",          2, 0, 20, 0.1, &rho);
    new ParamDouble(g, "precision_rho", sqrt(-2*log(0.05)), 1, 10, 1, &precision_rho);
    new ParamBool  (g, "st_normalize", false, &st_normalize);
    new ParamBool  (g, "ds_squared",   true, &ds_squared);
    new ParamInt   (g, "etf_N",        3, 0, 10, 1, &etf_N);

    g = new ParamGroup(this, "bilateral_filter");
    new ParamChoice(g, "type",       "xy", "full|xy|oa|fbl", &filter_type);
    new ParamInt   (g, "n_e",         1, 0, 100, 1, &n_e);
    new ParamInt   (g, "n_a",         4, 0, 100, 1, &n_a);
    new ParamDouble(g, "sigma_dg",    3, 0, 20, 0.05, &sigma_dg);
    new ParamDouble(g, "sigma_dt",    3, 0, 20, 0.05, &sigma_dt);
    new ParamDouble(g, "sigma_rg",    4.25, 0, 100, 0.05, &sigma_rg);
    new ParamDouble(g, "sigma_rt",    4.25, 0, 100, 0.05, &sigma_rt);
    new ParamDouble(g, "bf_alpha",    0, 0, 10000, 1, &bf_alpha);
    new ParamDouble(g, "precision_g", 2, 1, 10, 1, &precision_g);
    new ParamDouble(g, "precision_t", 2, 1, 10, 1, &precision_t);

    g = new ParamGroup(this, "dog");
    ParamGroup *dog = g;
    new ParamChoice(g, "dog_type", "flow-based", "isotropic|flow-based|xdog", &dog_type);
    new ParamChoice(g, "dog_input", "L", "L|gray|nvac|apparent-gray", &dog_input);
    new ParamDouble(g, "sigma_e", 1, 0, 20, 0.005, &sigma_e);
    new ParamDouble(g, "dog_k", 1.6, 0, 100, 0.05, &dog_k);
    new ParamDouble(g, "precision_e", 3, 1, 5, 0.1, &precision_e);
    new ParamDouble(g, "sigma_m", 3, 0, 20, 1, &sigma_m);
    new ParamDouble(g, "precision_m", 2, 1, 5, 0.1, &precision_m);
    new ParamDouble(g, "step_m", 1, 0.01, 2, 0.1, &step_m);
    new ParamDouble(g, "tau0",     0.99, 0, 2, 0.005, &tau0);
    new ParamDouble(g, "tau1",     0, 0, 2, 0.005, &tau1);
    new ParamDouble(g, "epsilon", 0, -10, 10, 0.005, &epsilon);
    new ParamDouble(g, "phi_e",   2, 0, 1000, 0.1, &phi_e);

    new ParamChoice(g, "dog_fgauss", "euler", "euler|rk2-nn|rk2|rk4", &dog_fgauss);
    new ParamBool  (g, "dog_fgauss_adaptive", true, &dog_fgauss_adaptive);
    new ParamDouble(g, "dog_fgauss_max", 22.5, 0.0, 90.0, 1, &dog_fgauss_max);
    new ParamInt   (g, "dog_N", 1, 0, 100, 1, &dog_N);
    new ParamChoice(g, "dog_blend", "multiply", "multiply|unsharp|shock|", &dog_blend);

    g = new ParamGroup(dog, "apparent_gray");
    new ParamInt   (g, "ag_N",   4, 1, 10, 1, &ag_N);
    new ParamDouble(g, "ag_k",   0.5, 0, 2, 0.01, &ag_k);
    new ParamDouble(g, "ag_p",   0.5, 0, 2, 0.01, &ag_p);

    g = new ParamGroup(this, "quantization", true, &quantization);
    new ParamChoice(g, "quant_type", "adaptive", "fixed|adaptive", &quant_type);
    new ParamInt   (g, "nbins", 8, 1, 255, 1, &nbins);
    new ParamDouble(g, "phi_q", 2, 0, 100, 0.025, &phi_q);
    new ParamDouble(g, "lambda_delta", 0, 0, 100, 1, &lambda_delta);
    new ParamDouble(g, "omega_delta", 2, 0, 100, 1, &omega_delta);
    new ParamDouble(g, "lambda_phi", 0.9, 0, 100, 1, &lambda_phi);
    new ParamDouble(g, "omega_phi", 1.6, 0, 100, 1, &omega_phi);

    g = new ParamGroup(this, "warp_sharp", false, &warp_sharp);
    new ParamDouble(g, "sigma_w", 1.5, 0, 20, 1, &sigma_w);
    new ParamDouble(g, "precision_w", 2, 1, 5, 0.1, &precision_w);
    new ParamDouble(g, "phi_w", 2.7, 0, 100, 0.025, &phi_w);

    g = new ParamGroup(this, "final_smooth", true, &final_smooth);
    new ParamChoice(g, "type", "flow-based", "3x3|5x5|flow-based", &final_type);
    new ParamDouble(g, "sigma_f", 1.0, 0, 10, 1, &sigma_f);

    g = new ParamGroup(this, "debug", false, &debug);
    new ParamBool(g, "draw_flow", false, &draw_flow);
}


void FlowAbs::process() {
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
    if (input_gamma == "linear-rgb") {
        src = linear2srgb(src);
    }
    if (auto_levels) {
        src = hist_auto_levels(src, auto_levels_threshold);
    }
    publish("src", src);

    if (noise) {
        gpu_image n = noise_normal(src.w(), src.h(), 0, variance);
        //if (half) n = gpu_set(n, 0, 0, 0, I.w()/2, I.h());
        publish("noise", n);
        src = src + n.convert(FMT_FLOAT3);
        publish("src-n", src);
    }

    gpu_image lab = rgb2lab(src);

    gpu_image st;
    if (st_type == "central-diff") {
        st = st_central_diff(rgb2gray(src));
        st = gauss_filter_xy(st, rho, precision_rho);
    }
    else if (st_type == "sobel-rgb") {
        st = st_sobel(src);
        st = gauss_filter_xy(st, rho, precision_rho);
    }
    else if (st_type == "sobel-lab") {
        st = st_sobel(lab);
        st = gauss_filter_xy(st, rho, precision_rho);
    }
    else if (st_type == "sobel-L") {
        st = st_sobel(shuffle(lab,0));
        st = gauss_filter_xy(st, rho, precision_rho);
    }
    else  if (st_type == "scharr-rgb") {
        st = st_scharr_3x3(src);
        st = gauss_filter_xy(st, rho, precision_rho);
    }
    else if (st_type == "etf-full") {
        gpu_image tfm = etf_full(rgb2gray(src), rho, etf_N);
        st = st_from_tangent(tfm);
    }
    else if (st_type == "etf-xy") {
        gpu_image tfm = etf_xy(rgb2gray(src), rho, etf_N);
        st = st_from_tangent(tfm);
    }
    else if (st_type == "gaussian-deriv") {
        st = st_gaussian(rgb2gray(src), sqrtf(0.433f*0.433f + rho*rho), precision_rho);
    }
    else {
        OZ_X() << "unsupported!";
    }

    gpu_image tf = st_to_tangent(st);
    publish("tf", tf);

    /*{
        gpu_image noise = noise_fast(tf.w(), tf.h(), 1);
        gpu_image flow = fgauss_filter(noise, tf, 6);
        flow = hist_eq(flow);
        flow = fgauss_filter(flow, tf, 1);
        publish("flow", flow);
    }*/

    gpu_image img = lab;
    gpu_image Ie = img;
    gpu_image Ia = img;
    int N = std::max(n_e, n_a);

    for (int i = 0; i < N; ++i) {
        if (filter_type == "oa") {
            gpu_image lfm = st_lfm(st, bf_alpha);
            img = oabf_1d(img, lfm, sigma_dg, sigma_rg, false, true, true, precision_g);
            img = oabf_1d(img, lfm, sigma_dt, sigma_rt, true, true, true, precision_t);
        } else if (filter_type == "xy") {
            img = bilateral_filter_xy(img, sigma_dg, sigma_rg);
        } else if (filter_type == "fbl") {
            gpu_image lfm = st_lfm(st, bf_alpha);
            img = oabf_1d(img, lfm, sigma_dg, sigma_rg, false, true, true, precision_g);
            img = stbf2_filter(img, st, sigma_dt, sigma_rt, precision_t, 90.0f, false, true, true, 2, 1);
        } else {
            img = bilateral_filter(img, sigma_dg, sigma_rg);
        }
        if (i == (n_e - 1)) Ie = img;
        if (i == (n_a - 1)) Ia = img;
    }

    gpu_image L;
    if (output != "fill") {
        if (dog_input == "gray") {
            L = rgb2gray(lab2rgb(Ie)) * 100;
        } else if (dog_input == "nvac") {
            L = rgb2nvac(lab2rgb(Ie));
        } else if (dog_input == "apparent-gray") {
            std::vector<float> k(ag_N, ag_k);
            L = apparent_gray(lab2rgb(Ie), ag_N, &k[0], ag_p);
        } else {
            L = shuffle(Ie, 0);
        }
        publish("L", L / 100);

        if (dog_type == "flow-based") {
            L = gradient_dog( L, tf, sigma_e, dog_k*sigma_e, tau0, tau1, precision_e );
            publish("dog-p1", dog_colorize(L));

            if (dog_fgauss == "euler") {
                L = fgauss_filter( L, tf, sigma_m, precision_m, step_m );
            } else if (dog_fgauss == "rk2-nn") {
                L = stgauss2_filter(L, st, sigma_m, dog_fgauss_max, dog_fgauss_adaptive, false, false, 2, step_m);
            } else if (dog_fgauss == "rk2") {
                L = stgauss2_filter(L, st, sigma_m, dog_fgauss_max, dog_fgauss_adaptive, true, true, 2, step_m);
            } else if (dog_fgauss == "rk4") {
                L = stgauss2_filter(L, st, sigma_m, dog_fgauss_max, dog_fgauss_adaptive, true, true, 4, step_m);
            }

            publish("dog-p2", dog_colorize(L));
            L = dog_threshold_tanh(L, epsilon, phi_e);
        } else if (dog_type == "isotropic") {
            L = wog_dog( L, sigma_e, dog_k*sigma_e, tau0, phi_e, epsilon, precision_e );
        }
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

    if (output == "edges") {
        img = gray2rgb(L);
    } else if (output == "fill+edges") {
        img = blend_intensity(img, L, BLEND_MULTIPLY);
    }
    if (input_gamma == "linear-rgb") {
        img = srgb2linear(img);
    }

    if (warp_sharp) {
        img = wog_warp_sharp(img, sigma_w, precision_w, phi_w);
    }

    if (final_smooth) {
       if (final_type == "3x3")
           img = gauss_filter_3x3(img);
       else if (final_type == "5x5")
           img = gauss_filter_5x5(img);
       else {
            img = fgauss_filter( img, tf, sigma_f);
       }
    }

    publish("$result", img);
}


void FlowAbs::draw(ImageView *view, QPainter &p, int pass) {
    QRect aR = p.clipBoundingRect().toAlignedRect().intersected(view->image().rect());
    Module::draw(view, p, pass);
    if (!debug) return;

    if (draw_flow && view->zoom() > 5) {
        double px = draw_pt2px(p);
        p.setPen(QPen(Qt::darkBlue, 1.5*px));
        cpu_image tf = publishedImage("tf");
        for (int j = aR.top(); j <= aR.bottom(); ++j) {
            for (int i = aR.left(); i <= aR.right(); ++i) {
                float2 t = tf.at<float2>(i, j);
                QPointF q(i+0.5, j+0.5);
                QPointF v(0.45 * t.x, 0.45 * t.y);
                p.drawLine(q-v, q+v);
            }
        }
    }
}
