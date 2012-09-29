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
#include "testring.h"
#include <oz/test_pattern.h>
#include <oz/st.h>
#include <oz/st_util.h>
#include <oz/stgauss.h>
#include <oz/noise.h>
#include <oz/gauss.h>
#include <oz/hist.h>
#include <oz/minmax.h>
#include <oz/colormap.h>
#include <oz/colormap_util.h>
#include <oz/ds.h>
#include <oz/rst.h>
#include <oz/etf.h>
#include <oz/color.h>
#include <oz/shuffle.h>
#include <oz/make.h>
#include <oz/blend.h>
#include <oz/resize.h>
#include <oz/resample.h>
#include <oz/blit.h>
#include <oz/pyr.h>
#include <oz/qimage.h>
#include <oz/grad.h>
using namespace oz;


TestRing::TestRing() {
    ParamGroup *g;

    g = new ParamGroup(this, "test image");
    new ParamChoice(g, "image_type", "jaehne", "zone-plate|jaehne|knutsson", &image_type);
    new ParamBool  (g, "image_8bit", false, &image_8bit);
    new ParamInt(g, "width", 512, 1, 4096, 1, &width);
    new ParamInt(g, "height", 512, 1, 4096, 1, &height);
    new ParamBool(g, "sRGB", true, &sRGB);
    new ParamBool  (g, "equalize", false, &equalize);
    new ParamDouble(g, "g0", 1, 0, 100, 0.01, &g0);
    new ParamDouble(g, "km", 0.9, 0, 4096, 0.01, &km);
    new ParamDouble(g, "rm", 240, 0, 4096, 0.25, &rm);
    new ParamDouble(g, "w",  8, 0, 4096, 0.25, &w);
    new ParamBool  (g, "inverted", false, &inverted);
    new ParamBool  (g, "st_8bit", false, &st_8bit);
    new ParamDouble(g, "st_scale",  1, 0, 100, 0.1, &st_scale);
    new ParamDouble(g, "st_gamma",  1, 0, 100, 0.01, &st_gamma);
    new ParamChoice(g, "st_resample", "none", "none|box|cubic|gaussian|pyr5x5", &st_resample);
    new ParamBool  (g, "st_liegr", false, &st_liegr);
    new ParamDouble(g, "hist_max",  1000, 0, 1e38, 1000, &hist_max);
    new ParamBool  (g, "use_grad", false, &use_grad);
    new ParamDouble(g, "grad_scale",  1, 0, 1e38, 0.05, &grad_scale);

    g = new ParamGroup(this, "noise");
    new ParamDouble(g, "variance",  0.001, 0, 1, 0.0005, &variance);

    g = new ParamGroup(this, "structure tensor");
    new ParamChoice(g, "gradient", "scharr-3x3", "central-diff|sobel|scharr-3x3|scharr-5x5|gaussian|gaussian-x2|rst|ds-axis|etf-full|etf-gaussian|etf-xy|grad-sobel|grad-scharr-3x3|grad-scharr-5x5|grad-gaussian|multi-scale", &gradient);
    new ParamDouble(g, "pre_blur",  0, 0, 20, 0.25, &pre_blur);
    new ParamDouble(g, "sigma",  0.433, 0, 20, 0.01, &sigma);
    new ParamDouble(g, "precision_sigma",  5, 1, 10, 0.25, &precision_sigma);
    new ParamDouble(g, "rho",  0, 0, 20, 0.1, &rho);
    new ParamDouble(g, "rho8",  0, 0, 20, 0.1, &rho8);
    new ParamDouble(g, "precision_rho",  3, 1, 10, 0.25, &precision_rho);
    new ParamDouble(g, "m",  1, 0, 20, 0.05, &m);
    new ParamBool  (g, "st_normalize", false, &st_normalize);
    new ParamBool  (g, "ds_squared", true, &ds_squared);
    new ParamInt   (g, "etf_N", 3, 0, 10, 1, &etf_N);
    new ParamInt   (g, "moa_mode", 1, 0, 8, 1, &moa_mode);
    new ParamChoice(g, "pyr_down_mode", "gaussian", "box|triangle|bell|quadratic|quadratic-approx|quadratic-mix|bspline|lanczos2|lanczos3|blackman|cubic|catrom|mitchell|gaussian", &pyrdownMode);

    g = new ParamGroup(this, "debug");
    new ParamDouble(g, "scale",  0.1, 0, 100000, 0.25, &scale);
    new ParamBool  (g, "draw_scale",  false, &draw_scale);
    new ParamBool  (g, "draw_st",  false, &draw_st);
}


void TestRing::mergeHalfs(const QString& c, const QString& a, const QString& b) {
    gpu_image ga = publishedImage(a).gpu().convert(FMT_FLOAT3);
    gpu_image gb = publishedImage(b).copy(ga.w()/2,0, ga.w()-1, ga.h()-1).gpu().convert(FMT_FLOAT3);

    gpu_image m(ga.w()+10, ga.h(), FMT_FLOAT3);
    m.fill(make_float3(1,1,1), 0, 0, m.w(), m.h());

    blit(m, 0, 0, ga, 0,0, ga.w()/2, ga.h());
    blit(m, ga.w()/2+10, 0, gb, 0, 0, gb.w(), gb.h());
    publish(c, m);
}


void TestRing::mergeQuads(const QString& r, const QString& a, const QString& b, const QString& c, const QString& d) {
    int w = publishedImage(a).w();
    int h = publishedImage(a).h();
    int w2 =  w / 2;
    int h2 =  h / 2;

    gpu_image ga = publishedImage(a).copy(0, 0, w2-1, h2-1).gpu().convert(FMT_FLOAT3);
    gpu_image gb = publishedImage(b).copy(w2, 0, w-1, h2-1).gpu().convert(FMT_FLOAT3);
    gpu_image gc = publishedImage(d).copy(w2, h2, w-1, h-1).gpu().convert(FMT_FLOAT3);
    gpu_image gd = publishedImage(c).copy(0, h2, w2-1, h-1).gpu().convert(FMT_FLOAT3);

    gpu_image m(w+8, h+8, FMT_FLOAT3);
    m.fill(make_float3(1,1,1), 0, 0, m.w(), m.h());

    blit(m, 0, 0, ga, 0, 0, w2, h2);
    blit(m, w2+8, 0, gb, 0, 0, w2, h2);
    blit(m, w2+8, h2+8, gc, 0, 0, w2, h2);
    blit(m, 0, h2+8, gd, 0, 0, w2, h2);
    publish(r, m);
}


void TestRing::process() {
    gpu_image R[2];
    if (image_type == "jaehne") {
        R[0] = test_jaenich_ring(width, height, g0, km, rm, w);
        R[1] = R[0];
    } else if (image_type == "zone-plate") {
        R[0] = test_zoneplate(width, height, g0, km, rm, w, inverted);
        R[1] = test_zoneplate(width, height, g0, km, rm, w, !inverted);
    } else if (image_type == "knutsson") {
        R[0] = test_knutsson_ring();
        if ((width != 512) || (height != 512)) {
            R[0] = resample(R[0], width, height, RESAMPLE_LANCZOS3);
        }
        R[1] = R[0];
    }

    if (!noise.is_valid() || (noise.size() != R[0].size()) || (noise_var != variance)) {
        noise = noise_normal(R[0].w(), R[0].h(), 0, variance);
        noise_var = variance;
    }
    gpu_image n = noise;
    publish("noise", n);

    for (int k = 0; k < 2; ++k) {
        gpu_image I = shuffle(R[k], 0);
        if (equalize) I = hist_eq(I);
        if (image_8bit) I = I.convert(FMT_UCHAR).convert(FMT_FLOAT);

        gpu_image mask = shuffle(R[k], 1);

        gpu_image In = I + noise;
        I = gauss_filter(I, pre_blur, 3);
        In = gauss_filter(In, pre_blur, 3);

        process2(QString("i%1:").arg(k), I, mask);
        process2(QString("n%1:").arg(k), In, mask);
    }

    //mergeHalfs("I", "i:I", "n:I");
    //mergeHalfs("error", "i:error", "n:error");

    //mergeHalfs("I", "i0:I", "i1:I");
    //mergeHalfs("error", "i0:error", "i1:error");

    mergeQuads("I", "i0:I", "i1:I", "n0:I", "n1:I");
    mergeQuads("error", "i0:error", "i1:error", "n0:error", "n1:error");

    /*{
        QImage L = to_qimage(publishedImage("I"));
        QImage R = to_qimage(publishedImage("error"));
        QImage H1 = to_qimage(publishedImage("i:diff"));
        QImage H2 = to_qimage(publishedImage("n:diff"));
        QImage I(L.width() + R.width() + 20 + H1.width(), qMax(qMax(L.height(), R.height()),H1.height()+H2.height()), QImage::Format_RGB32);
        I.fill(Qt::white);
        QPainter p(&I);
        p.drawImage(0, 0, L);
        p.drawImage(L.width()+10, 0, R);
        p.drawImage(L.width()+R.width()+20, 0, H1);
        p.drawImage(L.width()+R.width()+20, H1.height(), H2);
        publish("error2", I);
    }*/
}


void TestRing::process2(const QString& prefix, const gpu_image& I, const gpu_image& mask) {
    publish(prefix+"I", sRGB? linear2srgb(I) : I);

    gpu_image Ir = I;
    if (st_resample == "cubic")
        Ir = resample(I, I.w()*2, I.h()*2, RESAMPLE_CUBIC);
    else if (st_resample == "pyr5x5")
        Ir = pyrup_gauss5x5(I);

    gpu_image st, st_x2;
    gpu_image tm;
    if (gradient == "multi-scale") {
        OZ_X() << "!";
        //st = gpu_st_multi_scale(I, 5, rho, (gpu_resample_mode_t)pyrdownMode, st_normalize? 1 : 0, moa_mode);
    }
    else if (gradient == "central-diff") {
        st = st_central_diff(Ir);
        st = gauss_filter_xy(st, rho, precision_rho);
    }
    else if (gradient == "sobel") {
        st = st_sobel(Ir);
        st = gauss_filter_xy(st, rho, precision_rho);
    }
    else if (gradient == "scharr-3x3") {
        st = st_scharr_3x3(Ir, 0, st_normalize);
        st = gauss_filter_xy(st, rho, precision_rho);
    }
    else if (gradient == "scharr-5x5") {
        st = st_scharr_5x5(Ir, 0, st_normalize);
        st = gauss_filter_xy(st, rho, precision_rho);
    }
    else if (gradient == "gaussian") {
        st = st_gaussian(Ir, sigma, precision_sigma);
        st = gauss_filter_xy(st, rho, precision_rho);
    }
    else if (gradient == "gaussian-x2") {
        st = st_gaussian_x2(I, sigma, precision_sigma, false);
        st_x2 = st;
    }
    else if (gradient == "rst") {
        st = rst_scharr(Ir, rho, m);
    }
    else if (gradient == "ds-axis") {
        st = ds_scharr_3x3(Ir, rho, st_normalize, ds_squared);
    }
    else if (gradient == "etf-full") {
        tm = etf_full(Ir, rho, etf_N, precision_rho);
        st = st_from_tangent(tm);
    }
    else if (gradient == "etf-gaussian") {
        tm = etf_full(Ir, rho, etf_N, precision_rho, true);
        st = st_from_tangent(tm);
    }
    else if (gradient == "etf-xy") {
        tm = etf_xy(Ir, rho, etf_N, precision_rho);
        st = st_from_tangent(tm);
    }
    else if (gradient == "grad-sobel") {
        tm = grad_sobel(Ir, false);
        st = st_from_gradient(tm);
    }
    else if (gradient == "grad-scharr-3x3") {
        tm = grad_scharr_3x3(Ir, false);
        st = st_from_gradient(tm);
    }
    else if (gradient == "grad-scharr-5x5") {
        tm = grad_scharr_5x5(Ir, false);
        st = st_from_gradient(tm);
    }
    else if (gradient == "grad-gaussian") {
        tm = grad_gaussian(Ir, sigma, precision_sigma, false);
        st = st_from_gradient(tm);
        st = gauss_filter_xy(st, rho, precision_rho);
    }

    if ((st_resample == "none") && (gradient == "gaussian-x2")) {
        st = resize_half(st);
    }
    else if ((st_resample == "box") && (gradient == "gaussian-x2")) {
        st = resize_half(st);
        st = gauss_filter_xy(st, rho, precision_rho);

    }
    else  if (st_resample == "cubic") {
        st = resample(st, I.w(), I.h(), RESAMPLE_CUBIC);
    }
    else if (st_resample == "pyr5x5") {
        st = pyrdown_gauss5x5(st);
    }
    else if (st_resample == "gaussian") {
        st = resample_gaussian(st, I.w(), I.h(), rho, precision_rho );
    }

    if (st_liegr) st = st_log(st);
    if (st_gamma != 1.0) st = st_pow(st, st_gamma);
    if (st_scale != 1.0) st = st * st_scale;

    {
        gpu_image E = shuffle(st, 0);
        gpu_image F = shuffle(st, 2);
        gpu_image G = shuffle(st, 1);
        publish(prefix+"E", colormap_jet2(E));
        publish(prefix+"F", colormap_jet2(F));
        publish(prefix+"G", colormap_jet2(G));
        std::vector<int> H;
        H = hist(E, 512, -0.5f, 0.5f);
        publishHistogram(prefix+"H-E", H, hist_max);
        H = hist(F, 512, -0.5f, 0.5f);
        publishHistogram(prefix+"H-F", H, hist_max);
        H = hist(G, 512, -0.5f, 0.5f);
        publishHistogram(prefix+"H-G", H, hist_max);
    }

    if (st_8bit) {
        st = st + 0.5f;
        st = st.convert(FMT_UCHAR3).convert(FMT_FLOAT3);
        st = gauss_filter_xy(st, rho8, precision_rho);
        st = st.convert(FMT_UCHAR3).convert(FMT_FLOAT3);
        st = st - 0.5f;
    } else {
        st = gauss_filter_xy(st, rho8, precision_rho);
    }

    {
        gpu_image E = shuffle(st, 0);
        gpu_image F = shuffle(st, 2);
        gpu_image G = shuffle(st, 1);
        publish(prefix+"E2", E);
        publish(prefix+"F2", colormap_jet2(F));
        publish(prefix+"G2", linear2srgb(G));
        std::vector<int> H;
        H = hist(E, 512, -0.5f, 0.5f);
        publishHistogram(prefix+"H-E2", H, hist_max);
        H = hist(F, 512, -0.5f, 0.5f);
        publishHistogram(prefix+"H-F2", H, hist_max);
        H = hist(G, 512, -0.5f, 0.5f);
        publishHistogram(prefix+"H-G2", H, hist_max);
    }

    if (st_scale != 1.0) st = st / st_scale;
    if (st_gamma != 1.0) st = st_pow(st, 1.0f / st_gamma);
    if (st_liegr) st = st_exp(st);

    {
        /*
        gpu_image n = noise_normal(st.w(), st.h(), 0, 1);
        gpu_image lic = stgauss_filter(n, st, 6, 22.5f, false);
        lic = hist_eq(lic);
        lic = stgauss_filter(lic, st, 1.5, 22.5f, false);
        lic = linear2srgb(lic);

        gpu_image a = make(lic, lic, lic, mask);
        gpu_image b = gpu_image(lic.w(), lic.h(), make_float4(0.5f, 0.5f, 0.5f, 1.0f));
        a = blend(b, a, BLEND_NORMAL);

        publish(prefix+"flow", a);
        */
    }

    if (tm.is_valid()) {
        gpu_image gx = shuffle(tm, 0);
        gpu_image gy = shuffle(tm, 1);
        gx = 0.5f/grad_scale*gx+0.5f;
        gy = 0.5f/grad_scale*gy+0.5f;
        publish("gx", sRGB? linear2srgb(gx) : gx);
        publish("gy", sRGB? linear2srgb(gy) : gy);
    }

    publish(prefix+"st", st);
    gpu_image angle;
    if (tm.is_valid() && use_grad) {
        qDebug("grad_angle");
        angle = grad_angle(tm, true);
    } else {
        qDebug("st_angle");
        angle = st_angle(st);
    }

    gpu_image A = testring_xy_angle(I.w(), I.h());

    publish(prefix+"angle", 0.5f * angle / CUDART_PI_F + 0.5f);
    publish(prefix+"angle-xy", 0.5f * A / CUDART_PI_F + 0.5f);
    {
        if ((gradient == "gaussian-x2") && (st_resample == "none")) {
            gpu_image angle = st_angle(st_x2);
            gpu_image ed = testring_diff_angle(angle);

            gpu_image e = testring_jet(abs(ed), scale);
            e = make(e, resize_x2(mask));
            gpu_image b = gpu_image(e.w(), e.h(), make_float4(0.735f, 0.735f, 0.735f, 1.0f));
            e = blend(b, e, BLEND_NORMAL);

            //QImage err = to_qimage(e);
            //int ew = err.width();
            //int eh = err.height();
            //publish(prefix+"error", err.copy((ew-2*width)/2, (eh-2*height)/2, 2*width, 2*height));
            publish(prefix+"error", e);
        } else {
            gpu_image ed = testring_diff_angle(angle);
            std::vector<float> edpdf = pdf(ed, 256, -scale, scale);
            publish(prefix+"diff", edpdf);

            gpu_image e = testring_jet(abs(ed), scale);
            e = make(e, mask);
            gpu_image b = gpu_image(e.w(), e.h(), make_float4(0.735f, 0.735f, 0.735f, 1.0f));
            e = blend(b, e, BLEND_NORMAL);

            //QImage err = to_qimage(e);
            //int ew = err.width();
            //int eh = err.height();
            //publish(prefix+"error", err.copy((ew-width)/2, (eh-height)/2, width, height));
            publish(prefix+"error", e);
        }
    }
}



void TestRing::dragBegin(ImageView *view, QMouseEvent *e) {
    //QPointF p = view->view2image(QPointF(e->pos()));
    //cpu_image st = getImage("st");
    //if (st.is_valid()) {
    //    float3 g = st.at<float3>(p.x(), p.y());
    //}
}


static void drawST(const cpu_image& st, ImageView *view, QPainter &p, const QRect& R) {
    if (!st.is_valid())
        return;

    for (int j = R.top(); j <= R.bottom(); ++j) {
        for (int i = R.left(); i <= R.right(); ++i) {
            float3 g = st.at<float3>(i, j);
            float2 t = st2tangent(g);
            QPointF q(i+0.5, j+0.5);
            QPointF v(0.45 * t.x, 0.45 * t.y);
            p.drawLine(q-v, q+v);
        }
    }
}


void TestRing::draw(ImageView *view, QPainter &p, int pass) {
    Module::draw(view, p, pass);

    QRectF R = p.clipBoundingRect();
    QRect aR = R.toAlignedRect().intersected(view->image().rect());

    if (draw_st && (view->zoom() >= 2)) {
        p.save();
        p.setPen(Qt::blue);
        cpu_image st = publishedImage("i0:st");
        drawST(st, view, p, aR);
        p.restore();
    }

    if (draw_scale) {
        QImage map(256,1,QImage::Format_RGB32);
        for (int i = 0; i < 256; ++i) {
            float3 c = colormap_jet(1.0f * i / 255.0f);
            int b = (int)(c.x * 255);
            int g = (int)(c.y * 255);
            int r = (int)(c.z * 255);
            map.setPixel(i, 0, qRgb(r,g,b));
        }

        QFont font("Arial");
        font.setPixelSize(20);
        p.setFont(font);
        p.setPen(Qt::black);
        int y = height - 20;
        p.drawText(QRect(0, y, 40, 20), Qt::AlignCenter, QString("0") + QChar((short)0x00B0) );
        p.drawImage(QRect(40, 3+y+2, width-80, 12), map, QRect(0,0,256,1));
        p.drawText(QRect(width-40, y, 40,20), Qt::AlignCenter, QString("%1").arg(scale) + QChar((short)0x00B0) );
    }
}
