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
#include "msakftest.h"
#include <oz/st.h>
#include <oz/msst.h>
#include <oz/color.h>
#include <oz/noise.h>
#include <oz/resize.h>
#include <oz/resample.h>
#include <oz/stgauss.h>
#include <oz/hist.h>
#include <oz/color.h>
#include <oz/minmax.h>
#include <oz/gkf_kernel.h>
#include <oz/msakf.h>
#include <oz/shuffle.h>
#include <oz/make.h>
#include <oz/ssia.h>
#include <oz/dog.h>
#include <oz/gpu_timer.h>
#include <oz/colormap.h>
#include <oz/colormap_util.h>
using namespace oz;


QImage renderhPyr(const std::vector<gpu_image>& P, int orientation=-1, bool drawScale=false) {
    if (orientation < 0) {
        orientation = P[0].w() < P[0].h();
    }

    int w, h;
    if (orientation == 0) {
        w = P[0].w() + P[1].w() + 10;
        h = P[0].h();
    } else {
        w = P[0].w();
        h = P[0].h() + P[1].h() + 10;
    }

    QImage img(w, h, QImage::Format_RGB32);
    img.fill(0xffffffff);
    QPainter p(&img);

    int x = 0;
    int y = 0;
    for (unsigned i = 0; i < P.size(); ++i) {

        QImage I = to_qimage(P[i]);
        p.drawImage(x, y, I);

        if ((i + orientation) & 1) {
            y += P[i].h() + 10;
        } else {
            x += P[i].w() + 10;
        }
    }

    if (drawScale) {
        QImage map(256,1,QImage::Format_RGB32);
        for (int i = 0; i < 256; ++i) {
            float3 c = colormap_jet(1.0f * i / 255.0f);
            int b = (int)(c.x * 255);
            int g = (int)(c.y * 255);
            int r = (int)(c.z * 255);
            map.setPixel(i, 0, qRgb(r,g,b));
        }

        QFont font("Arial");
        font.setPixelSize(10);
        p.setFont(font);
        p.setPen(Qt::black);
        int x,y,sw;
        if (orientation == 0) {
            sw = P[1].w();
            x = P[0].w() + 5;
            y = h - 20;
        } else {
            sw = w / 2 - 10;
            x = w / 2;
            y = h - 20;
        }
        p.drawText(QRect(x, y, 28,20), Qt::AlignRight | Qt::AlignVCenter, "0%");
        p.drawImage(QRect(x+30, y+2, sw-60, 16), map, QRect(0,0,256,1));
        p.drawText(QRect(x+sw-28, y, 28,20), Qt::AlignLeft | Qt::AlignVCenter, "100%");
    }

    return img;
}


void MsAkfTest::publishPyr(const QString& key, const std::vector<gpu_image>& P) {
    int w = 0;
    int h = 0;
    for (unsigned i = 0; i < P.size(); ++i) {
        w += P[i].w();
        h = std::max<int>(h, P[i].h());
    }
    w += 2 * ((int)P.size() - 1);

    QImage img(w, h, QImage::Format_RGB32);
    img.fill(0);
    QPainter p(&img);
    w = 0;
    for (unsigned i = 0; i < P.size(); ++i) {
        QImage I = to_qimage(P[i]);
        p.drawImage(w, 0, I);
        w += I.width() + 2;
    }
    publish(key, img);
}


///////////////////////////////////////////////////////////////////////////////


MsAkfTest::MsAkfTest() {
    ParamGroup *g;
    g = new ParamGroup(this, "auto_levels", false, &auto_levels);
    new ParamDouble(g, "threshold", 0.1, 0, 100, 0.05, &auto_levels_threshold);

    g = new ParamGroup(this, "noise", false, &noise);
    new ParamDouble(g, "variance",  0.01, 0, 1, 0.005, &variance);

    g = new ParamGroup(this, "pyramid");
    new ParamChoice(g, "pyr_down", "lanczos3", "box|triangle|bell|quadratic|quadratic-approx|quadratic-mix|bspline|lanczos2|lanczos3|blackman|cubic|catrom|mitchell|gaussian|kaiser", &pyr_down);
    new ParamChoice(g, "pyr_up", "triangle", "box|triangle|bell|quadratic|quadratic-approx|quadratic-mix|bspline|lanczos2|lanczos3|blackman|cubic|catrom|mitchell|gaussian|kaiser", &pyr_up);
    new ParamChoice(g, "pyr_factor", "2", "2|sqrt(2)", &pyr_factor);

    g = new ParamGroup(this, "structure tensor");
    new ParamBool  (g, "st_enable_ms", true, &st_enable_ms);
    new ParamDouble(g, "rho", 2, 0, 10, 0.25f, &rho);
    new ParamDouble(g, "st_epsilon", 0, 0, 100, 0.025f, &st_epsilon);
    new ParamChoice(g, "moa", "squared", "squared|sqrt|squared2", (int*)&moa);
    new ParamBool  (g, "st_indep_prop", false, &st_indep_prop);

    g = new ParamGroup(this, "options");
    new ParamInt(g, "radius", 6, 0, 50, 1, &radius);
    new ParamDouble(g, "q", 8, 1, 16, 1, &q);
    new ParamDouble(g, "alpha", 1.0, 0, 1000, 1, &alpha);
    new ParamDouble(g, "threshold", 0.02, 0, 10, 1e-3, &threshold);
    new ParamDouble(g, "scale", 0.5, 0, 100, 0.1, &scale);
    new ParamDouble(g, "damping", 1.25, 0, 1000, 0.025, &damping);
    new ParamDouble(g, "vthresh", 0, 0, 1, 0.025, &vthresh);
    new ParamInt(g, "prop_mode", 0, 0, 2, 1, &prop_mode);
    new ParamBool(this, "shock", false, &shock);

    g = new ParamGroup(this, "debug", false, &debug);
    new ParamInt(g, "orientation", -1, -1,1,1, &orientation);
    new ParamInt(g, "showLevel", -1, -1, 5, 1, &showLevel);
    new ParamBool(g, "draw_scale", false, &draw_scale);

    krnl4 = circshift(gkf_create_kernel4(32, 0.33f, 2.5f, 8), 16, 16);
}


gpu_image stvis(const gpu_image& st) {
    gpu_image n = noise_fast(st.w(), st.h(), 1);
    gpu_image lic = stgauss_filter(n, st, 6, 22.5f, false);
    return hist_eq(lic);
}


void MsAkfTest::process() {
    gpu_image src = gpuInput0();
    if (auto_levels) {
        src = hist_auto_levels(src, auto_levels_threshold);
    }
    if (noise) {
        src = add_gaussian_noise(src, 0, variance);
    }
    publish("src", src);

    gpu_timer tt;
    std::vector<gpu_image> P;
    {
        gpu_image cur = src.convert(FMT_FLOAT4);
        while (P.size() < 5) {
            P.push_back(cur);
            if ((cur.w() <= 1) || (cur.h() <= 1)) break;

            int w, h;
            if (pyr_factor == "2") {
                w = (cur.w()+1)/2;
                h = (cur.h()+1)/2;
            } else {
                w = (cur.w()+1)/sqrtf(2);
                h = (cur.h()+1)/sqrtf(2);
            }

            cur = resample(cur, w, h, (resample_mode_t)pyr_down);
        }
    }

    std::vector<gpu_image> A;
    A.resize(P.size());

    std::vector<gpu_image> ST;
    ST.resize(P.size());

    std::vector<gpu_image> V;
    V.resize(P.size() -1);

    for (int k = (int)P.size() - 1; k >= 0; --k) {
        if (k < (int)P.size() - 1) {
            gpu_image Ak1;
            Ak1 = resample(A[k+1], P[k].w(), P[k].h(), (resample_mode_t)pyr_up);

            gpu_image Vk1 = shuffle(Ak1, 3);
            Vk1 = adjust(Vk1, scale * powf(damping, k), -vthresh);
            V[k] = Vk1;

            if (k + 1 == showLevel) {
                Ak1 = gpu_image(Ak1.w(), Ak1.h(), make_float4(1,0,0,1));
            }

            if (shock && (k == 0)) {
                gpu_image dog = dog_filter(rgb2gray(A[k+1]), 1, 1.6f, 1, 0);
                dog = resample(dog, P[k].w(), P[k].h(), (resample_mode_t)pyr_up);
                Ak1 = ssia_shock(Ak1, dog);
            }

            A[k] = msakf_propagate(P[k], Ak1, Vk1, 1);

            if (st_indep_prop)
                ST[k] = st_scharr_3x3(P[k].convert(FMT_FLOAT3), rho);
            else
                ST[k] = st_scharr_3x3(A[k].convert(FMT_FLOAT3), rho);
            if (st_enable_ms) {
                gpu_image STk1 = resample(ST[k+1], P[k].w(), P[k].h(), (resample_mode_t)pyr_up);
                ST[k] = st_moa_merge(ST[k], STk1, st_epsilon, moa);
           }

            A[k] = msakf_single(A[k], ST[k], krnl4, radius, q, alpha, threshold);
        } else {
            ST[k] = st_scharr_3x3(P[k].convert(FMT_FLOAT3), rho);
            A[k] = msakf_single(P[k], ST[k], krnl4, radius, q, alpha, threshold);
        }
    }

    if (debug) {
        std::vector<gpu_image> PP;
        for (unsigned i = 0; i < P.size(); ++i) PP.push_back(P[i].convert(FMT_FLOAT3));

        std::vector<gpu_image> AA;
        for (unsigned i = 0; i < A.size(); ++i) AA.push_back(A[i].convert(FMT_FLOAT3));

        std::vector<gpu_image> VC;
        for (unsigned i = 0; i < V.size(); ++i) VC.push_back(colormap_jet(V[i]));

        std::vector<gpu_image> MOA;
        for (unsigned i = 0; i < ST.size(); ++i) MOA.push_back(colormap_jet(st_moa(ST[i], moa)));

        std::vector<gpu_image> VF;
        for (unsigned i = 0; i < V.size(); ++i) {
            VF.push_back(stvis(ST[i]));
        }

        publish("P", renderhPyr(PP, orientation));
        publish("V", renderhPyr(V, orientation));
        publish("VC", renderhPyr(VC, orientation, true));
        publish("VF", renderhPyr(VF, orientation));
        publish("A", renderhPyr(AA, orientation));
        publish("MOA", renderhPyr(MOA, orientation));
    } else {
        qDebug() << "time" << tt.elapsed_time();
    }

    publish("$result", A[0].convert(FMT_FLOAT3));

    /*
    gpu_image<float4> mst = gpu_st_multi_scale(src, 5, rho);
    publish("mst-0", stvis(ST[0]));
    publish("mst-1", stvis(mst));

    int H[256];
    gpu_hist_256(gpu_32f_to_8u(gpu_rgb2gray(A[0])), H);
    publishHistogram("H-result", H, 256);

    gpu_hist_256(gpu_32f_to_8u(gpu_rgb2gray(src)), H);
    publishHistogram("H-src", H, 256);
    */
}


/*
void MsAkfTest::dragBegin(ImageView *view, QMouseEvent *e) {
    QPointF p = view->view2image(QPointF(e->pos()));
    cpu_image<float4> st = getImage<float4>("Sk1");
    float4 g = st(p.x(), p.y());

    cpu_image<float> Gx = getImage<float>("gx");
    float gx = Gx(p.x(), p.y());
    cpu_image<float> Gy = getImage<float>("gy");
    float gy = Gy(p.x(), p.y());

    qDebug() << p << g.x << g.y << g.z << gx << gy;
}
*/


void MsAkfTest::draw(ImageView *view, QPainter &p, int pass) {
    QRect aR = p.clipBoundingRect().toAlignedRect().intersected(view->image().rect());
    Module::draw(view, p, pass);
}
