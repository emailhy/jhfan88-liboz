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
#include "akftest.h"
#include <oz/color.h>
#include <oz/colormap.h>
#include <oz/make.h>
#include <oz/noise.h>
#include <oz/kuwahara.h>
#include <oz/gkf_kernel.h>
#include <oz/akf.h>
#include <oz/akf_opt.h>
#include <oz/akf_opt2.h>
#include <oz/akf_opt3.h>
#include <oz/akf_opt4.h>
#include <oz/polyakf_opt.h>
#include <oz/polyakf_opt2.h>
#include <oz/hist.h>
#include <oz/st.h>
#include <oz/st_util.h>
#include <oz/gpu_timer.h>
#include <oz/qpainter_draw.h>
#include <oz/gauss.h>
#include <oz/norm.h>
#include <oz/minmax.h>
#include <oz/shuffle.h>
#include <oz/blit.h>
#include <oz/progress.h>
#include <oz/csv.h>
#include <cfloat>
using namespace oz;


AnisotropicKuwahara::AnisotropicKuwahara() {
    noise_variance = -1;
    krnl_smoothing = 0;

    ParamGroup *g;
    g = new ParamGroup(this, "auto_levels", false, &auto_levels);
    new ParamDouble(g, "threshold", 0.1, 0, 100, 0.05, &auto_levels_threshold);

    g = new ParamGroup(this, "noise", false, &noise);
    new ParamDouble(g, "variance",  0.01, 0, 1, 0.005, &variance);

    g = new ParamGroup(this, "options");
    new ParamChoice(g, "method", "tex8", "tex4|tex4-opt|tex8|tex8-opt|tex8-opt2|tex8-opt3|tex8-opt4|poly4|poly8|poly8-opt", &method);
    new ParamInt   (g, "N", 1, 1, 10, 1, &N);
    new ParamDouble(g, "rho", 2, 0, 10, 1, &rho);
    new ParamBool  (g, "disable_st", false, &disable_st);
    new ParamDouble(g, "smoothing", 33.33f, 0, 100, 0.1, &smoothing);
    new ParamDouble(g, "precision", 2.5, 0, 5, 0.25f, &precision);
    new ParamInt   (g, "radius", 7, 1, 200, 1, &radius);
    new ParamDouble(g, "q", 8, 1, 16, 1, &q);
    new ParamDouble(g, "alpha", 1.0, 0, 1000, 1, &alpha);
    new ParamDouble(g, "threshold", 1e-4, 0, 1, 1e-4, &threshold);
    new ParamDouble(g, "a_star", 0, 0, 1, 0.05, &a_star);

    g = new ParamGroup(this, "debug", false, &debug);
    pdebug_x = new ParamInt(g, "debug_x", 0, 0, 4096, 1, &debug_x);
    pdebug_y = new ParamInt(g, "debug_y", 0, 0, 4096, 1, &debug_y);
    new ParamDouble(g, "scale_w", 1, 0, FLT_MAX, 1, &scale_w);
    new ParamBool(g, "draw_st", false, &draw_st);
    new ParamBool(g, "draw_filter", false, &draw_filter);
    new ParamBool(g, "show_weights", false, &show_weights);
    new ParamBool(g, "draw_isotropic", false, &draw_isotropic);
    new ParamBool(g, "draw_origin", false, &draw_origin);

    g = new ParamGroup(this, "benchmark");
    new ParamInt (g, "benchmark_n", 500, 1, 10000, 100, &benchmark_n );
}


void AnisotropicKuwahara::process() {
    if ((krnl_smoothing != smoothing) || (krnl_precision != precision)) {
        krnl_smoothing = smoothing;
        krnl_precision = precision;
        krnl41 = circshift(gkf_create_kernel1(32, smoothing / 100.0f, precision, 4), 16, 16);
        krnl44 = circshift(gkf_create_kernel4(32, smoothing / 100.0f, precision, 4), 16, 16);
        krnl81 = circshift(gkf_create_kernel1(32, smoothing / 100.0f, precision, 8), 16, 16);
        krnl84 = circshift(gkf_create_kernel4(32, smoothing / 100.0f, precision, 8), 16, 16);
        krnl8x2 = circshift(gkf_create_kernel8x2(32, smoothing / 100.0f, precision), 16, 16);
    }

    gpu_image src = gpuInput0();

    if (noise && ((noise_variance != variance) || noise_img.size() != src.size())) {
        noise_variance = variance;
        gpu_image n = noise_normal(src.w(), src.h(), 0, variance);
        noise_img = make(n, n, n);
    }

    publish("krnl41", krnl41);
    publish("krnl44.x", shuffle(krnl44, 0));
    publish("krnl44", vstack_channel(krnl44, 10));

    if (auto_levels) {
        src = hist_auto_levels(src, auto_levels_threshold);
    }
    if (noise) {
        src = src + noise_img;
    }
    publish("$src", src);

    gpu_timer tt;
    gpu_image st;
    if (!disable_st)
        st = st_scharr_3x3(src, rho);
    else
        st = gpu_image(src.w(), src.h(), make_float3(0,1,0));
    if (debug) publish("st", st);

    gpu_image dst = src;
    for (int k = 0; k < N; ++k) {
        if (method == "tex4") {
            dst = akf_filter(dst, st, krnl41, radius, q, alpha, threshold, a_star, 4);
        }
        else if (method == "tex4-opt") {
            dst = akf_opt_filter(dst, st, krnl44, radius, q, alpha, threshold, 4);
        }
        else if (method == "tex8") {
            if (debug && (k == N-1)) {
                gpu_image w(8 * src.w(), src.h(), FMT_FLOAT);
                dst = akf_filter(dst, st, krnl81, radius, q, alpha, threshold, a_star, 8, &w);
                publish("w", w);
            } else {
                dst = akf_filter(dst, st, krnl81, radius, q, alpha, threshold, a_star, 8);
            }
        }
        else if (method == "tex8-opt") {
            dst = akf_opt_filter(dst, st, krnl84, radius, q, alpha, threshold, 8);
        }
        else if (method == "tex8-opt2") {
            dst = akf_opt_filter2(dst, st, krnl8x2, radius, q, alpha, threshold);
        }
        else if (method == "tex8-opt3") {
            dst = akf_opt_filter3(dst, st, krnl8x2, radius, q, alpha, threshold);
        }
        else if (method == "tex8-opt4") {
            dst = akf_opt_filter4(dst, st, krnl84, radius, q, alpha, threshold, a_star);
        }
        else if (method == "poly4") {
            dst = polyakf_opt_filter(dst, st, radius, q, alpha, threshold, 2.0/radius, 0.84f, 4);
        }
        else if (method == "poly8") {
            dst = polyakf_opt_filter(dst, st, radius, q, alpha, threshold, 2.0/radius, 3.77f, 8);
        }
        else if (method == "poly8-opt") {
            dst = polyakf_opt_filter2(dst, st, radius, q, alpha, threshold, 2.0/radius, 3.77f);
        }
    }
    if (!debug) {
        qDebug() << "time" << tt.elapsed_time();
    }
    publish("$result", dst);
}



void AnisotropicKuwahara::benchmark() {
    std::string name = QInputDialog::getText(NULL, "Benchmark", "ID:").toStdString();

    QStringList M;
    M << "tex4" << "tex4-opt" << "tex8" << "tex8-opt" << "tex8-opt2" << "poly4" << "poly8" << "poly8-opt";

    gpu_image src = gpuInput0();
    gpu_timer tt;
    gpu_image st;
    if (!disable_st)
        st = st_scharr_3x3(src, rho);
    else
        st = gpu_image(src.w(), src.h(), make_float3(0,1,0));

    progress_t progress(0, 1, 0, M.size());
    for (int m = 0; m < M.size(); m++) {
        progress_t progress(m, m+1, 2, radius+1);
        cpu_image T(radius, benchmark_n, FMT_FLOAT);
        T.clear();

        for (int r = 2; r <= radius; ++r) {
            progress_t progress(r, r+1, 0, benchmark_n-1);
            double t0 = 0;
            for (int n = 0; n < benchmark_n; ++n) {
                if (!progress(n)) return;

                tt.reset();
                gpu_image dst;
                if (M[m] == "tex4") {
                    dst = akf_filter(src, st, krnl41, r, q, alpha, threshold, a_star, 4);
                }
                else if (M[m] == "tex4-opt") {
                    dst = akf_opt_filter(src, st, krnl44, r, q, alpha, threshold, 4);
                }
                else if (M[m] == "tex8") {
                    dst = akf_filter(src, st, krnl81, r, q, alpha, threshold, a_star, 8);
                }
                else if (M[m] == "tex8-opt") {
                    dst = akf_opt_filter(src, st, krnl84, r, q, alpha, threshold, 8);
                }
                else if (M[m] == "tex8-opt2") {
                    dst = akf_opt_filter2(src, st, krnl8x2, r, q, alpha, threshold);
                }
                else if (M[m] == "poly4") {
                    dst = polyakf_opt_filter(src, st, r, q, alpha, threshold, 2.0/r, 0.84f, 4);
                }
                else if (M[m] == "poly8") {
                    dst = polyakf_opt_filter(src, st, r, q, alpha, threshold, 2.0/r, 3.77f, 8);
                }
                else if (M[m] == "poly8-opt") {
                    dst = polyakf_opt_filter2(src, st, r, q, alpha, threshold, 2.0/r, 3.77f);
                }
                double t = tt.elapsed_time();
                T.at<float>(r,n) = t;
                t0 += t / benchmark_n;
            }

            qDebug() << M[m] << "radius=" << r << t0 << "ms";
        }
        csv_write(T, "akf-" + name + "_" + M[m].toStdString() + ".csv");
    }
}


void AnisotropicKuwahara::test() {
    gpu_image src = gpuInput0();
    gpu_timer tt;
    gpu_image st;
    if (!disable_st)
        st = st_scharr_3x3(src, rho);
    else
        st = gpu_image(src.w(), src.h(), make_float3(0,1,0));

    //progress_t progress(r, r+1, 0, benchmark_n-1);
    double t0 = 0;
    for (int n = 0; n < benchmark_n; ++n) {
        //if (!progress(n)) return;
        tt.reset();
        gpu_image dst;
        if (method == "tex4") {
            dst = akf_filter(src, st, krnl41, radius, q, alpha, threshold, a_star, 4);
        }
        else if (method == "tex4-opt") {
            dst = akf_opt_filter(src, st, krnl44, radius, q, alpha, threshold, 4);
        }
        else if (method == "tex8") {
            dst = akf_filter(src, st, krnl81, radius, q, alpha, threshold, a_star, 8);
        }
        else if (method == "tex8-opt") {
            dst = akf_opt_filter(src, st, krnl84, radius, q, alpha, threshold, 8);
        }
        else if (method == "tex8-opt2") {
            dst = akf_opt_filter2(src, st, krnl8x2, radius, q, alpha, threshold);
        }
        else if (method == "poly4") {
            dst = polyakf_opt_filter(src, st, radius, q, alpha, threshold, 2.0/radius, 0.84f, 4);
        }
        else if (method == "poly8") {
            dst = polyakf_opt_filter(src, st, radius, q, alpha, threshold, 2.0/radius, 3.77f, 8);
        }
        else if (method == "poly8-opt") {
            dst = polyakf_opt_filter2(src, st, radius, q, alpha, threshold, 2.0/radius, 3.77f);
        }
        double t = tt.elapsed_time();
        t0 += t / benchmark_n;
    }
    qDebug() << method << "radius=" << radius << t0 << "ms";
}



void AnisotropicKuwahara::dragBegin(ImageView *view, QMouseEvent *e) {
    QPointF p = view->view2image(e->pos());
    pdebug_x->setValue(p.x());
    pdebug_y->setValue(p.y());
}


void AnisotropicKuwahara::draw(ImageView *view, QPainter &p, int pass) {
    QRect aR = p.clipBoundingRect().toAlignedRect().intersected(view->image().rect());
    Module::draw(view, p, pass);

    if (debug && draw_st && (view->zoom() >= 2)) {
        p.save();
        p.setPen(Qt::blue);
        cpu_image st = publishedImage("st");
        draw_minor_eigenvector_field(p, st, aR);
        p.restore();
    }

    if (debug && (draw_origin || draw_filter)) {
        int ix = debug_x;
        int iy = debug_y;
        QPointF m_p(ix + 0.5f, iy + 0.5f);

        if (draw_origin) {
            p.save();
            p.setPen(QPen(Qt::red, 0.5f, Qt::SolidLine, Qt::RoundCap));
            p.drawPoint(m_p);
            p.restore();
        }

        if (draw_filter && (view->zoom() >= 2)) {
            p.save();
            p.translate(m_p);

            cpu_image W = publishedImage("w");
            double w[8];
            float max_w = 0;
            for (int i = 0; i < 8; ++i) {
                w[i] = W.at<float>(8*ix+i, iy);
                if (w[i] > max_w) max_w = w[i];
            }
            for (int i = 0; i < 8; ++i) {
                w[i] /= max_w;
            }

            float3 st = publishedImage("st").at<float3>(ix,iy);
            float3 t = st2tA(st, a_star);
            int N = 8;
            float phi = atan2f(t.y, t.x);
            float A = t.z;
            float a = radius * clamp((alpha + A) / alpha, 0.1f, 2.0f);
            float b = radius * clamp(alpha / (alpha + A), 0.1f, 2.0f);

            p.rotate(180.0f * phi / CUDART_PI_F);

            for (int k = 0; k < N; ++k ) {
                float ksize = show_weights? w[k] : 1;
                p.save();
                p.scale(a, b);
                QPainterPath path;
                path.moveTo(0, 0);
                path.arcTo(-ksize,-ksize, 2*ksize, 2*ksize, 360.0*k/N-180.0/N, 360.0/N);
                path.closeSubpath();
                if ((w[k] > 0.2) || (show_weights /*&& (ksize >= 0.25 / radius)*/))
                    p.fillPath(path, QColor(255,128,128));
                p.restore();
            }

            p.drawEllipse(QPointF(0,0), a, b);
            for (int k = 0; k < N; ++k ) {
                float om = -k * 2.0 * CUDART_PI_F / N;
                om -= CUDART_PI_F / N;
                p.drawLine(QPointF(0,0), QPointF(cos(om) * a, sin(om) * b));
            }

            p.restore();
        }
    }
}


#if 0
AnisotropicKuwahara2::AnisotropicKuwahara2() {
    new ParamDouble(this, "rho", 2, 0, 10, 1, &rho);
    new ParamDouble(this, "zeta", 0.33, 0, 1, 0.01, &zeta);
    new ParamDouble(this, "eta", 3.77, 0, 10, 0.1, &eta);
    new ParamInt  (this, "radius", 7, 1, 32, 1, &radius);
    new ParamDouble(this, "q", 8, 1, 16, 1, &q);
    new ParamDouble(this, "alpha", 1.0, 0, 1000, 1, &alpha);
}


void AnisotropicKuwahara2::process() {
    gpu_image<float4> src = gpuInput0<float4>();
    gpu_timer tt;
    gpu_image<float4> tfm = gpu_st_tfm(gpu_st_scharr(src, rho));
    gpu_image<float4> dst = gpu_polynomial_akf_opt8(
        src,
        tfm,
        radius,
        q,
        alpha,
        zeta,
        eta
    );
    qDebug() << "time" << tt.get_elapsed_time();
    publish("$result", dst);
}


AnisotropicKuwaharaDiff::AnisotropicKuwaharaDiff() {
    jet_image = QImage(300,1, QImage::Format_RGB32);
    for (int i = 0; i < jet_image.width(); ++i) {
        float4 c = colormap_jet((float)i / (jet_image.width()-1));
        unsigned b = c.x * 255;
        unsigned g = c.y * 255;
        unsigned r = c.z * 255;
        jet_image.setPixel(i, 0, qRgb(r,g,b));
    }

    krnl_smoothing = 0;
    new ParamDouble(this, "rho", 2, 0, 10, 1, &rho);
    new ParamDouble(this, "smoothing", 33.33f, 0, 100, 0.01, &smoothing);
    new ParamDouble(this, "zeta", 0.33, 0, 1, 0.1, &zeta);
    new ParamDouble(this, "eta", 3.77, 0, 10, 0.1, &eta);
    new ParamInt   (this, "radius", 7, 1, 32, 1, &radius);
    new ParamDouble(this, "q", 8, 1, 16, 1, &q);
    new ParamDouble(this, "alpha", 1.0, 0, 1000, 1, &alpha);
    new ParamDouble(this, "diff", 10, 0, 100, 1, &diff);
}


void AnisotropicKuwaharaDiff::process() {
    if (krnl_smoothing != smoothing) {
        krnl_smoothing = smoothing;
        krnl = gpu_gkf_create_kernel4(32, smoothing / 100.0f);
    }
    gpu_image<float4> src = gpuInput0<float4>();
    gpu_image<float4> tfm = gpu_st_tfm(gpu_st_scharr(src, rho));

    gpu_image<float4> tex = gpu_kernel_based_akf_opt8(
        src,
        tfm,
        krnl,
        radius,
        q,
        alpha
    );
    publish("tex", tex);

    gpu_image<float4> poly = gpu_polynomial_akf_opt8(
        src,
        tfm,
        radius,
        q,
        alpha,
        zeta,
        eta
    );
    publish("poly", poly);

    cpu_image<uchar4> tmp = gpu_32f_to_8u(gpu_colormap_diff(tex, poly, diff / 100.0f)).cpu();
    QImage D = QImage((const uchar*)tmp.ptr(), tmp.w(), tmp.h(), tmp.pitch(), QImage::Format_RGB32);

    {
        QPainter p(&D);
        int x = (D.width() - jet_image.width()) / 2;
        int y = D.height() - 20;
        for (int i = 0; i < 20; ++i) {
            p.drawImage(x, D.height()-1-i, jet_image);
        }
        p.setFont(QFont("Arial", 9, QFont::Bold));
        p.setPen(QPen(Qt::white));
        p.drawText(QRect(x-40, y, 40, 20), Qt::AlignCenter, "0%");
        p.drawText(QRect(x+jet_image.width(), y, 40, 20), Qt::AlignCenter, QString("%1%").arg((int)diff));
    }
    publish("$D", D);
}
#endif