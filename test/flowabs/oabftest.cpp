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
#include "oabftest.h"
#include <oz/color.h>
#include <oz/bilateral.h>
#include <oz/oabf.h>
#include <oz/oabf2.h>
#include <oz/gauss.h>
#include <oz/shuffle.h>
#include <oz/make.h>
#include <oz/st.h>
#include <oz/wog.h>
#include <oz/qpainter_draw.h>
#include <oz/stbf2.h>
#include <oz/stbf3.h>
#include <oz/stgauss3.h>
#include <oz/colormap.h>
#include <oz/minmax.h>
#include <oz/licint.h>
#include <oz/noise.h>
using namespace oz;


OaBfTest::OaBfTest() {
    noise_variance = -1;
    ParamGroup *g;

    g = new ParamGroup(this, "noise", false, &noise);
    new ParamDouble(g, "variance",  0.01, 0, 1, 0.005, &variance);

    new ParamInt   (this, "N", 4, 0, 50, 1, &N);

    g = new ParamGroup(this, "structure tensor");
    new ParamDouble(g, "rho", 2, 0, 10, 1, &rho);
    new ParamChoice(g, "st_colorspace", "CIELAB", "CIELAB|RGB", &st_colorspace);
    new ParamBool  (g, "st_recalc", false, &st_recalc);

    g = new ParamGroup(this, "bilateral filter");
    new ParamChoice(g, "method", "oabf", "full|xy|oabf2|fbl3|lic|fbl|oabf", &method);
    new ParamChoice(g, "bf_colorspace", "CIELAB", "CIELAB|RGB", &bf_colorspace);
    new ParamChoice(g, "order", "rk2", "euler|rk2", &order);
    new ParamDouble(g, "sigma_dg", 3, 0, 20, 1, &sigma_dg);
    new ParamDouble(g, "sigma_rg", 4.25, 0, 100, 1, &sigma_rg);
    new ParamDouble(g, "precision_g", 2, 1, 10, 1, &precision_g);
    new ParamDouble(g, "sigma_dt", 3, 0, 20, 1, &sigma_dt);
    new ParamDouble(g, "sigma_rt", 4.25, 0, 100, 1, &sigma_rt);
    new ParamDouble(g, "precision_t", 2, 1, 10, 1, &precision_t);
    new ParamBool  (g, "adaptive", false, &adaptive);
    new ParamBool  (g, "ustep", false, &ustep);
    new ParamChoice(g, "src_linear", "off", "off|on|last-only", &src_linear_ex);
    new ParamBool  (g, "st_linear", true, &st_linear);
    new ParamDouble(g, "step_size", 1, 0, 10, 0.1, &step_size);
    new ParamBool  (g, "nonuniform_sigma", false, &nonuniform_sigma);

    g = new ParamGroup(this, "quantization", true, &quantization);
    new ParamChoice(g, "quant_type", "adaptive", "fixed|adaptive", &quant_type);
    new ParamInt   (g, "nbins", 8, 1, 255, 1, &nbins);
    new ParamDouble(g, "phi_q", 2, 0, 100, 0.025, &phi_q);
    new ParamDouble(g, "lambda_delta", 0, 0, 100, 1, &lambda_delta);
    new ParamDouble(g, "omega_delta", 2, 0, 100, 1, &omega_delta);
    new ParamDouble(g, "lambda_phi", 0.9, 0, 100, 1, &lambda_phi);
    new ParamDouble(g, "omega_phi", 1.6, 0, 100, 1, &omega_phi);

    g = new ParamGroup(this, "debug", false, &debug);
    p_debug_x = new ParamDouble(g, "debug_x", 0, 0, 4096, 0.25, &debug_x);
    p_debug_y = new ParamDouble(g, "debug_y", 0, 0, 4096, 0.25, &debug_y);
    new ParamBool  (g, "draw_orientation", false, &draw_orientation);
    new ParamBool  (g, "draw_center", false, &draw_center);
    new ParamBool  (g, "draw_line_p1", false, &draw_line_p1);
    new ParamBool  (g, "draw_active_p1", false, &draw_active_p1);
    new ParamBool  (g, "draw_samples_p1", false, &draw_samples_p1);
    new ParamBool  (g, "draw_line_p2", false, &draw_line_p2);
    new ParamBool  (g, "draw_active_p2", false, &draw_active_p2);
    new ParamBool  (g, "draw_samples_p2", false, &draw_samples_p2);
    new ParamBool  (g, "draw_midpoint", false, &draw_midpoint);
    new ParamBool  (g, "show_plot", false, &show_plot);
}


void OaBfTest::process() {
    gpu_image src = gpuInput0();
    if (noise && ((noise_variance != variance) || noise_img.size() != src.size())) {
        noise_variance = variance;
        gpu_image n = noise_normal(src.w(), src.h(), 0, variance);
        noise_img = make(n, n, n);
    }
    if (noise) {
        src = src + noise_img;
    }
    publish("src", src);

    gpu_image img = (bf_colorspace == "CIELAB")? rgb2lab(src) : src;
    gpu_image st;
    gpu_image lfm;
    for (int k = 0; k < N; ++k) {
        bool src_linear = (src_linear_ex == "on") || ((k == N-1) && (src_linear_ex == "last-only"));

        double Ni = 1;
        if (nonuniform_sigma) {
            Ni = sqrt(3.0) * pow(2.0,N - k) / sqrt(pow(4.0, N) - 1);
            qDebug() << k << Ni;
        }

        if (method == "full") {
            img = bilateral_filter(img, Ni*sigma_dt, sigma_rt, precision_t);
        } else if (method == "xy") {
            img = bilateral_filter_xy(img, Ni*sigma_dt, sigma_rt, precision_t);
        } else {
            if ((k == 0) || st_recalc) {
                st = st_scharr_3x3((st_colorspace == "CIELAB")? img : src, rho);
                lfm = st_lfm(st);
                if (debug) publish("st", st);
                if (debug) publish("lfm", lfm);
            }

            if (method == "oabf2") {
                gpu_image tf = st_to_tangent(st);
                img = oabf2(img, tf, Ni*sigma_dg, sigma_rg, sigma_dt, sigma_rt, src_linear, precision_t);
            } else {
                img = oabf_1d(img, lfm, Ni*sigma_dg, sigma_rg, false, src_linear, ustep, precision_g);

                if (method == "oabf") {
                    img = oabf_1d(img, lfm, Ni*sigma_dt, sigma_rt, true, src_linear, ustep, precision_t);
                } else if (method == "fbl") {
                    img = stbf2_filter(img, st, Ni*sigma_dt, sigma_rt, precision_t, 90,
                                       adaptive, src_linear, st_linear, (order == "euler")? 1 : 2, step_size);
                } else  if (method == "fbl3") {
                    img = stbf3_filter_(img, st, Ni*sigma_dt, sigma_rt, precision_t, src_linear,
                                        st_linear, ustep, (order == "euler")? 1 : 2, step_size);
                } else if (method == "lic") {
                    gpu_image tf = st_to_tangent(st);
                    if (debug) publish("tf", tf);
                    img = licint_bf_flt(img, tf, Ni*sigma_dt, sigma_rt, precision_t);
                }
            }
        }
    }

    publish("L_jet", colormap_jet( normalize(shuffle(img,0))));

    if (quantization) {
        if (quant_type == "fixed")
            img = wog_luminance_quant( img, nbins, phi_q );
        else
            img = wog_luminance_quant( img, nbins, lambda_delta, omega_delta, lambda_phi, omega_phi );
    }

    if (bf_colorspace == "CIELAB") img = lab2rgb(img);
    publish("$result", img);
}


void OaBfTest::draw(ImageView *view, QPainter &p, int pass) {
    Module::draw(view, p, pass);
    if (!debug) return;

    QRectF R = p.clipBoundingRect();
    QRect aR = R.toAlignedRect().intersected(view->image().rect());
    double pt = view->pt2px(1);
    QPointF pos(floor(debug_x) + 0.5f, floor(debug_y) + 0.5f);

    if (draw_orientation && (view->zoom() > 3) && pass ) {
        p.save();
        p.setPen(QPen(Qt::gray, 0.25*pt));
        cpu_image st = publishedImage("st");
        draw_minor_eigenvector_field(p, st, aR);
        p.restore();
    }
    if (!pass) return;

    if (draw_line_p1 || draw_line_p2 || draw_samples_p1 || draw_samples_p2) {
        for (int tangential=0; tangential <= 1; ++tangential) {

            if (!tangential || (method == "oabf")) {
                cpu_image lfm = publishedImage("lfm");
                std::vector<float3> L = oabf_line(
                    pos.x(), pos.y(), lfm,
                    tangential? sigma_dt : sigma_dg,
                    tangential? sigma_rt : sigma_rg,
                    tangential,
                    tangential? precision_t : precision_g );

                int dir = oabf_sample_dir(pos.x(), pos.y(), lfm, tangential);
                QPolygonF pp;
                for (std::vector<float3>::iterator i = L.begin(); i != L.end(); ++i ) {
                    pp.push_back(QPointF(i->x, i->y));
                }

                if ((!tangential && draw_samples_p1) || (tangential && draw_samples_p2)) {
                    for (int i= 0; i < pp.size(); ++i) {
                        QPointF p1, p2;
                        if (dir == 1)  {
                            p1 = QPointF(floor(pp[i].x() - 0.5f) + 0.5f, pp[i].y());
                            p2 = QPointF(floor(pp[i].x() + 0.5f) + 0.5f, pp[i].y());
                        } else {
                            p1 = QPointF(pp[i].x(), floor(pp[i].y() - 0.5f) + 0.5f);
                            p2 = QPointF(pp[i].x(), floor(pp[i].y() + 0.5f) + 0.5f);
                        }

                        QPainterPath path;
                        path.addEllipse(p1, 2.45*pt, 2.45*pt);
                        path.addEllipse(p2, 2.45*pt, 2.45*pt);
                        p.fillPath(path, Qt::yellow);

                        p.setPen(QPen(Qt::yellow, 0.5*pt, Qt::SolidLine, Qt::RoundCap));
                        p.drawLine(p1, p2);
                    }
                }

                if ((!tangential && draw_line_p1) || (tangential && draw_line_p2)) {

                    if ((!tangential && draw_active_p1) || (tangential && draw_active_p2))
                        p.setPen(QPen(Qt::black, 0.5*pt, Qt::SolidLine, Qt::RoundCap));
                    else
                        p.setPen(QPen(Qt::darkBlue, 0.5*pt, Qt::SolidLine, Qt::RoundCap));
                    p.drawPolyline(pp);
                    for (int i= 0; i < pp.size(); ++i) {
                        QPainterPath path;
                        path.addEllipse(pp[i], 1.45*pt, 1.45*pt);
                        p.fillPath(path, Qt::black);
                    }
                }
            } else if (method == "lic") {
                if (tangential) {
                    cpu_image tf = publishedImage("tf");
                    std::vector<float3> L = licint_path(pos.x(), pos.y(), tf, sigma_dt, precision_t, draw_midpoint);

                    QPolygonF pp;
                    for (std::vector<float3>::iterator i = L.begin(); i != L.end(); ++i ) {
                        pp.push_back(QPointF(i->x, i->y));
                    }

                    if (draw_active_p2)
                        p.setPen(QPen(Qt::black, 0.5*pt, Qt::SolidLine, Qt::RoundCap));
                    else
                        p.setPen(QPen(Qt::darkBlue, 0.5*pt, Qt::SolidLine, Qt::RoundCap));

                    p.drawPolyline(pp);
                    for (int i= 0; i < pp.size(); ++i) {
                        QPainterPath path;
                        path.addEllipse(pp[i], 1.45*pt, 1.45*pt);
                        p.fillPath(path, Qt::black);
                    }
                }
            } else if (method == "fbl3") {
                cpu_image st = publishedImage("st");
                std::vector<float3> L = stgauss3_path_(pos.x(), pos.y(), st, sigma_dt,
                                                       st_linear, adaptive, ustep, (order == "euler")? 1 : 2, step_size);

                QPolygonF pp;
                for (std::vector<float3>::iterator i = L.begin(); i != L.end(); ++i ) {
                    pp.push_back(QPointF(i->x, i->y));
                }

                if (draw_samples_p2) {
                    for (int i= 0; i < pp.size(); ++i) {
                        if (!ustep) {
                            QPointF q[4];
                            q[0] = QPointF(floor(pp[i].x() - 0.5f) + 0.5f, floor(pp[i].y() - 0.5f) + 0.5f);
                            q[1] = QPointF(floor(pp[i].x() + 0.5f) + 0.5f, floor(pp[i].y() - 0.5f) + 0.5f);
                            q[2] = QPointF(floor(pp[i].x() - 0.5f) + 0.5f, floor(pp[i].y() + 0.5f) + 0.5f);
                            q[3] = QPointF(floor(pp[i].x() + 0.5f) + 0.5f, floor(pp[i].y() + 0.5f) + 0.5f);
                            QPainterPath path;
                            for (int j = 0; j < 4; ++j) {
                                path.addEllipse(q[j], 2.45*pt, 2.45*pt);
                            }
                            p.fillPath(path, Qt::yellow);
                            p.setPen(QPen(Qt::yellow, 0.5*pt, Qt::SolidLine, Qt::RoundCap));
                            for (int j = 0; j < 4; ++j) {
                                p.drawLine(pp[i], q[j]);
                            }
                        } else {
                            /*if ((fp.x() == 0.5f) && (fp.y() == 0.5f)) {
                                QPainterPath path;
                                path.addEllipse(pp[i], 2.45*pt, 2.45*pt);
                                p.fillPath(path, Qt::yellow);
                            } else*/ {
                                QPointF p1, p2;
                                #if 1
                                {
                                    float2 p = make_float2( pp[i].x(), pp[i].y() );
                                    p -= make_float2(0.5f, 0.5f);
                                    float2 ip = floor(p);
                                    float2 fp = p - ip;

                                    if (fp.x > 1e-4f) {
                                        float x = ip.x;
                                        float y = p.y;
                                        p1 = QPointF(x + 0.5f, y + 0.5f);
                                        p2 = QPointF(x + 1 + 0.5f, y + 0.5f);
                                    }
                                    else if (fp.y > 1e-4f) {
                                        float x = p.x;
                                        float y = ip.y;
                                        p1 = QPointF(x + 0.5f, y + 0.5f);
                                        p2 = QPointF(x + 0.5f, y + 1 + 0.5f);
                                    }
                                    else {
                                        p1 = p2 = pp[i];
                                    }

                                    /*
                                    if ((fp.x() == 0.5f) && (fp.y() == 0.5f)) {
                                        int xx = 0;
                                    }
                                    float2 p = make_float2(pp[i].x(), pp[i].y());
                                    float2 dp = 0.5f * make_float2((float)(fabs(fract(p.y - 0.5f)) < 1e-5f),
                                                                   (float)(fabs(fract(p.x - 0.5f)) < 1e-5f));

                                    p1 = QPointF(floor(pp[i].x() - dp.x) + 0.5f, floor(pp[i].y() - dp.y) + 0.5f);
                                    p2 = QPointF(floor(pp[i].x() + dp.x) + 0.5f, floor(pp[i].y() + dp.y) + 0.5f);
                                    */
                                }
                                #else
                                {
                                    if ((fp.x() == 0.5f)) {
                                        p1 = QPointF(pp[i].x(), floor(pp[i].y() - 0.5f) + 0.5f);
                                        p2 = QPointF(pp[i].x(), floor(pp[i].y() + 0.5f) + 0.5f);
                                    } else {
                                        p1 = QPointF(floor(pp[i].x() - 0.5f) + 0.5f, pp[i].y());
                                        p2 = QPointF(floor(pp[i].x() + 0.5f) + 0.5f, pp[i].y());
                                    }
                                }
                                #endif
                                QPainterPath path;
                                path.addEllipse(p1, 2.45*pt, 2.45*pt);
                                path.addEllipse(p2, 2.45*pt, 2.45*pt);
                                p.fillPath(path, Qt::yellow);

                                p.setPen(QPen(Qt::yellow, 0.5*pt, Qt::SolidLine, Qt::RoundCap));
                                p.drawLine(p1, p2);
                            }
                        }
                    }


                }

                if (draw_line_p2) {
                    if ((!tangential && draw_active_p1) || (tangential && draw_active_p2))
                        p.setPen(QPen(Qt::black, 0.5*pt, Qt::SolidLine, Qt::RoundCap));
                    else
                        p.setPen(QPen(Qt::darkBlue, 0.5*pt, Qt::SolidLine, Qt::RoundCap));

                    p.drawPolyline(pp);
                    draw_points(p, pp, 1.45*pt, Qt::black);
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
                            for (int i = 0; i < (int)L.size(); ++i) {
                                if (sign * L[i].z >= 0) pp.push_back(L[i]);
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
        }
    }

    if (draw_center) {
        QPainterPath path;
        path.addEllipse(pos, (1.45-0.25/2)*pt, (1.45-0.25/2)*pt);
        p.fillPath(path, Qt::red);
        p.strokePath(path, QPen(Qt::black, 0.5*pt, Qt::SolidLine, Qt::RoundCap));
    }
}


void OaBfTest::dragBegin(ImageView *view, QMouseEvent* e) {
    QPointF pos = view->view2image(e->pos());
    p_debug_x->setValue(pos.x());
    p_debug_y->setValue(pos.y());
    qDebug() << QString("clicked (%1, %2)").arg(pos.x()).arg(pos.y());
}
