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
#include "ivacef.h"
#include <oz/ivacef.h>
#include <oz/color.h>
#include <oz/st.h>
#include <oz/gauss.h>
#include <oz/grad.h>
#include <oz/stgauss2.h>
#include <oz/stgauss3.h>
#include <oz/laplace_eq.h>
#include <oz/convpyr.h>
#include <oz/qpainter_draw.h>
#include <oz/stintrk2.h>
#include <oz/cpu_sampler.h>
#include <oz/dog.h>
#include <oz/hist.h>
#include <oz/noise.h>
#include <oz/make.h>
#include <oz/licint.h>
#include <oz/filter_path.h>
#include <algorithm>
using namespace oz;


IVACEF::IVACEF() {
    noise_variance = -1;

    ParamGroup *g;
    new ParamInt   (this, "N", 5, 1, 100, 1, &N);

    g = new ParamGroup(this, "auto_levels", false, &auto_levels);
    new ParamDouble(g, "threshold", 0.1, 0, 100, 0.05, &auto_levels_threshold);
    g = new ParamGroup(this, "noise", false, &noise);
    new ParamDouble(g, "variance",  0.01, 0, 1, 0.005, &variance);

    g = new ParamGroup(this, "structure tensor");
    new ParamChoice(g, "method", "laplace", "none|original|laplace|convpyr", &method);
    new ParamDouble(g, "sigma_d", 1.0, 0.0, 10.0, 0.05, &sigma_d);
    new ParamDouble(g, "tau_r", 0.002, 0.0, 1.0, 0.001, &tau_r);
    new ParamChoice(g, "stencil", "stencil4", "stencil4|stencil8|stencil12|stencil20", (int*)&stencil);
    new ParamChoice(g, "upfilt", "linear-fast", "nearest|linear-fast|linear||cubic-fast|cubic", (int*)&upfilt);
    new ParamInt   (g, "v2", 1, 0, 1e6, 10, &v2);

    g = new ParamGroup(this, "adaptive smoothing");
    new ParamDouble(g, "sigma_t", 6.0, 0.0, 1000.0, 1, &sigma_t);
    new ParamChoice(g, "order", "rk2", "euler|rk2|lic", &order);
    new ParamDouble(g, "step_size", 1, 0.01, 10.0, 0.1, &step_size);
    new ParamBool  (g, "adaptive", false, &adaptive);
    new ParamBool  (g, "src_linear", true, &src_linear);
    new ParamBool  (g, "st_linear", true, &st_linear);
    new ParamBool  (g, "ustep", false, &ustep);

    g = new ParamGroup(this, "shock filtering", true, &shock_filtering);
    new ParamDouble(g, "sigma_i", 0.0, 0.0, 10.0, 0.25, &sigma_i);
    new ParamDouble(g, "sigma_g", 1.5, 0.0, 10.0, 0.25, &sigma_g);
    new ParamDouble(g, "r", 2, 0.0, 10.0, 0.25, &r);
    new ParamDouble(g, "tau_s", 0.005, -2, 2, 0.01, &tau_s);

    g = new ParamGroup(this, "post processing");
    new ParamDouble(g, "sigma_a", 1.5, 0.0, 10.0, 0.25, &sigma_a);

    g = new ParamGroup(this, "debug", false, &debug);
    p_debug_x = new ParamDouble(g, "debug_x", 0, 0, 4096, 0.25, &debug_x);
    p_debug_y = new ParamDouble(g, "debug_y", 0, 0, 4096, 0.25, &debug_y);
    new ParamBool  (g, "show gradients", false, &draw_gradients);
    new ParamBool  (g, "draw_orientation", false, &draw_orientation);
    new ParamBool  (g, "draw_streamline", false, &draw_streamline);
    new ParamBool  (g, "draw_streamline_linear", false, &draw_streamline_linear);
    new ParamBool  (g, "show_plot", false, &show_plot);
    new ParamBool  (g, "draw_midpoint", false, &draw_midpoint);
}


void IVACEF::process() {
    gpu_image img = gpuInput0().convert(FMT_FLOAT3);

    if (noise && ((noise_variance != variance) || noise_img.size() != img.size())) {
        noise_variance = variance;
        gpu_image n = noise_normal(img.w(), img.h(), 0, variance);
        noise_img = make(n, n, n);
    }

    if (auto_levels) {
        img = hist_auto_levels(img, auto_levels_threshold);
    }
    if (noise) {
        img = img + noise_img;
    }
    publish("$src", img);

    gpu_image st;
    if (debug) {
        gpu_image gray = rgb2gray(img);
        publish("src-gray", gray);
        publish("g", grad_sobel(gray, true));
    }

    for (int k = 0; k < N; ++k) {
        st = computeSt(img, st);
        if (debug) {
            publish("st", st);
        }
        if (order == "lic") {
            gpu_image tf = st_to_tangent(st);
            if (debug) publish("tf", tf);
            img = licint_gauss_flt(img, tf, sigma_t);
        } else {
            img = stgauss3_filter_(img, st, sigma_t, src_linear, st_linear, adaptive, (order=="rk2")? 2 : 1, step_size);
        }

        if (shock_filtering) {
            st = computeSt(img, st);
            gpu_image L = gauss_filter_xy(rgb2gray(img), sigma_i);
            gpu_image sign = ivacef_sign(L, st, sigma_g, tau_s);
            if (debug) publish("sign", dog_colorize(50 * sign));
            img = ivacef_shock(img, st, sign, r);
        }
    }

    img = stgauss3_filter_(img, st, sigma_a, true, true, false, 2, 1);
    publish("$result", img);
}


gpu_image IVACEF::computeSt( const gpu_image& src, const gpu_image& prev ) {
    gpu_image st;
    if (method == "none")
        st = st_scharr_3x3(src);
    else
        st = ivacef_sobel(src, prev, tau_r);
    if (!prev.is_valid()) {
        if (method == "original") {
            st = ivacef_relax(st, v2);
        } else if (method == "laplace") {
            st = leq_vcycle(st, v2, stencil, upfilt);
        } else if (method == "convpyr") {
            st = convpyr_boundary(st);
        }
    }
    st = st.convert(FMT_FLOAT3);
    return gauss_filter_xy(st, sigma_d);
}


void IVACEF::draw(ImageView *view, QPainter &p, int pass) {
    Module::draw(view, p, pass);
    if (!debug) return;

    QRectF R = p.clipBoundingRect();
    QRect aR = R.toAlignedRect().intersected(view->image().rect());
    QPointF pos(floor(debug_x) + 0.5f, floor(debug_y) + 0.5f);
    double pt = view->pt2px(1);

    if (draw_gradients && (view->zoom() > 3) && pass) {
        p.save();
        p.setPen(QPen(Qt::red, 0.25*pt));
        cpu_image g = publishedImage("g");
        draw_vector_field(p, g, aR, true);
        p.restore();
    }
    if (draw_orientation && (view->zoom() > 3) && pass) {
        p.save();
        p.setPen(QPen(Qt::blue, 0.25*pt));
        cpu_image g = publishedImage("g");
        draw_orientation_field(p, g, aR);
        p.restore();
    }

    if (draw_streamline && (view->zoom() > 3)) {
        cpu_image st = publishedImage("st");
        p.setPen(QPen(Qt::blue, 0.25*pt));
        draw_minor_eigenvector_field(p, st, aR);

        QPointF c = pos;
        std::vector<float3> path;
        if (order == "lic") {
            cpu_image tf = publishedImage("tf");
            path = licint_path( c.x(), c.y(), tf, sigma_t, 2, draw_midpoint);
        } else {
            path = stgauss3_path_( c.x(), c.y(), st, sigma_t, st_linear, adaptive, ustep, (order=="rk2")? 2 : 1, step_size);
        }

        QPolygonF P;
        for (int i = 0; i < (int)path.size(); ++i) {
            P.append(QPointF(path[i].x, path[i].y));
        }

        p.setPen(QPen(Qt::black, 0.5*pt, Qt::SolidLine, Qt::RoundCap));
        p.drawPolyline(P);
        draw_points(p, P, 1.45*pt, Qt::black);
        {
            QPainterPath path;
            path.addEllipse(pos, (1.45-0.25/2)*pt, (1.45-0.25/2)*pt);
            p.fillPath(path, Qt::red);
            p.strokePath(path, QPen(Qt::black, 0.5*pt, Qt::SolidLine, Qt::RoundCap));
        }

        if (show_plot) {
            QImage P(750, 400, QImage::Format_RGB32);
            {
                P.fill(Qt::white);
                QPainter p(&P);
                p.setWindow(-12,-20, 24, 40);
                p.drawLine(-100, 0, 100, 0);

                int sign = -1;
                do {
                    std::vector<float3> pp;
                    for (int i = 0; i < (int)path.size(); ++i) {
                        if (sign * path[i].z >= 0) pp.push_back(path[i]);
                    }
                    if (sign < 0)
                        std::reverse(pp.begin(), pp.end());

                    std::vector<float> ppx;
                    ppx.push_back(0);
                    float sum = 0;
                    for (int i = 1; i < (int)pp.size(); ++i) {
                        float2 dp = make_float2(pp[i].x, pp[i].y) - make_float2(pp[i-1].x, pp[i-1].y);
                        sum += length(dp);
                        ppx.push_back(sign * sum);
                    }

                    p.setPen(Qt::blue);
                    QPolygonF c;
                    c.push_back(QPointF(0,0));
                    for (int i = 1; i < (int)pp.size(); ++i) {
                        c.push_back(QPointF(
                            ppx[i],
                            pp[i].z
                        ));
                    }
                    p.drawPolyline(c);

                    p.setPen(Qt::black);
                    for (int i = 1; i < (int)pp.size(); ++i) {
                        p.drawLine(
                            QPointF(ppx[i], pp[i].z+0.1),
                            QPointF(ppx[i], pp[i].z-0.1)
                        );
                    }

                    sign *= -1;
                } while (sign > 0);
            }
            p.save();
            p.translate(2, -5);
            p.translate(pos.x(), pos.y());
            p.scale(1/view->zoom(), 1/view->zoom());
            p.drawImage(0, 0, P);
            p.restore();
        }
    }

    if (draw_streamline_linear && (view->zoom() > 3)) {
        cpu_image st = publishedImage("st");
        p.setPen(QPen(Qt::blue, 0.25*pt));
        draw_minor_eigenvector_field(p, st, aR);

        QPointF c = pos;
        std::vector<float3> path;
        path = stgauss3_path_( c.x(), c.y(), st, sigma_t, st_linear, adaptive, true, (order=="rk2")? 2 : 1, step_size);

        QPolygonF P;
        for (int i = 0; i < (int)path.size(); ++i) {
            P.append(QPointF(path[i].x, path[i].y));
        }

        p.setPen(QPen(Qt::red, 0.5*pt, Qt::SolidLine, Qt::RoundCap));
        p.drawPolyline(P);
        draw_points(p, P, 1.45*pt, Qt::red);
    }
}


void IVACEF::dragBegin(ImageView *view, QMouseEvent* e) {
    QPointF pos = view->view2image(e->pos());
    p_debug_x->setValue(pos.x());
    p_debug_y->setValue(pos.y());
    qDebug() << QString("clicked (%1, %2)").arg(pos.x()).arg(pos.y());
}

