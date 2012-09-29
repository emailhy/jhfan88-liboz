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
#include "testpattern.h"
#include <oz/test_pattern.h>
#include <oz/noise.h>
#include <oz/color.h>
#include <oz/st.h>
#include <oz/st_util.h>
#include <oz/gauss.h>
#include <oz/hist.h>
#include <oz/stgauss2.h>
#include <oz/grad.h>
#include <oz/minmax.h>
#include <oz/fft.h>
#include <oz/window.h>
#include <oz/colormap.h>
#include <oz/colormap_util.h>
#include <oz/qpainter_draw.h>
using namespace oz;


TestPattern::TestPattern() {
    nimg_variance = 0;
    ParamGroup *g;

    new ParamChoice(this, "pattern", "simple", "simple|image", &pattern);
    new ParamChoice(this, "function", "cos", "cos|sin|sinc", &function);

    g = new ParamGroup(this, "image");
    new ParamInt    (g, "width", 128, 1, 4096, 1, &width);
    new ParamInt    (g, "height", 128, 1, 4096, 1, &height);
    new ParamBool   (g, "sRGB", true, &sRGB);

    g = new ParamGroup(this, "noise", false, &noise);
    new ParamDouble (g, "variance",  0.01, 0, 1, 0.005, &variance);

    g = new ParamGroup(this, "structure tensor");
    new ParamChoice(g, "gradient", "scharr", "central-diff|scharr|gaussian-derivatives", &gradient);
    new ParamDouble(g, "rho",  0, 0, 20, 0.25, &rho);

    g = new ParamGroup(this, "simple");
    new ParamDouble (g, "phi",  22.5, 0, 359, 0.25, &phi);
    new ParamDouble (g, "phase",  8, 0, 1000, 0.25, &phase);
    new ParamDouble (g, "scale",  0.9, 0, 1, 0.025, &scale);

    g = new ParamGroup(this, "display");
    new ParamBool   (g, "draw_st", false, &draw_st);
    new ParamBool   (g, "draw_gf", false, &draw_gf);
    new ParamBool   (g, "draw_tf", false, &draw_tf);
    new ParamBool   (g, "draw_axis", false, &draw_axis);
    new ParamBool   (g, "draw_angle", false, &draw_angle);
    new ParamBool   (g, "draw_gradient", false, &draw_gradient);
    new ParamBool   (g, "draw_tangent", false, &draw_tangent);
    new ParamBool   (g, "draw_mini", false, &draw_mini);
    new ParamBool   (g, "draw_scale",  false, &draw_scale);
}


void TestPattern::process() {
    gpu_image I;
    if (pattern == "simple") {
        I = test_simple(width, height, phi, phase, scale, function);
    } else {
        I = rgb2gray(gpuInput0());
    }
    if (noise) {
        if (!nimg.is_valid() || (nimg.size() != I.size()) || (variance != nimg_variance)) {
            nimg = noise_normal(I.w(), I.h(), 0, variance);
            nimg_variance = variance;
        }
        publish("noise", nimg);
        I = I + nimg;
    }
    publish("$result", sRGB? linear2srgb(I) : I);

    gpu_image W = cosine_window(I.w(), I.h(), I.w() / 2);
    publish("W", W);
    gpu_image IW = I*W;
    publish("I*W", sRGB? linear2srgb(IW) : IW);

    gpu_image F;
    F = normalize(log_abs(fftshift(fft2(IW)) + 0.01f));
    publish("F-logabs", colormap_jet(F));

    F = normalize(log(abs(fftshift(fft2(IW))) + 1));
    publish("F-logabs2", colormap_jet(F));

    F = fftshift(abs(fft2(IW))) / sqrtf(I.w() * I.h());
    publish("F-norm", colormap_jet(F));

    F = fftshift(abs2(fft2(IW))) / sqrtf(I.w() * I.h());
    publish("F-norm2", colormap_jet(F));

    publish("st2", st_scharr_3x3(I, 2));

    gpu_image st;
    if (gradient == "central-diff") {
        st = st_central_diff(I);
        st = gauss_filter_xy(st, rho);
        publish("gf", grad_central_diff(I, true));
    }
    else if (gradient == "scharr") {
        st = st_scharr_3x3(I, rho);
        publish("gf", grad_scharr_3x3(I, true));
    } else {
        st = st_gaussian(I, sqrtf(0.433f*0.433f + rho*rho), 5);
        gpu_image b = gauss_filter_xy(I, rho);
        publish("blurred", sRGB? linear2srgb(b) : b);
        publish("gf", grad_gaussian(I, rho, true));
    }


    {
        gpu_image n = noise_normal(2*st.w(), 2*st.h(), 0, 1);
        gpu_image lic = stgauss2_filter(n, st, 12, 90.0f, false, true, true, 2, 1);
        lic = hist_eq(lic);
        lic = stgauss2_filter(lic, st, 1.5, 22.5f, false, true, true, 2, 1);
        publish("n", n);
        publish("flow", sRGB? linear2srgb(lic) : lic);
    }

    {
        gpu_image Is = gauss_filter_xy(I, rho);
        publish("I_rho", Is);
    }

    {
        gpu_image gf = grad_central_diff(I, true);
        publish("gf", gf);
    }
    publish("st", st);
}


void TestPattern::draw(ImageView *view, QPainter &p, int pass) {
    int layoutIndex = view->property("layoutIndex").toInt();

    QRectF R = p.clipBoundingRect();
    QRect aR = R.toAlignedRect().intersected(view->image().rect());
    QImage image = view->image();
    double px = draw_pt2px(p);
    double L = qMin(R.width(), R.height());

    if (layoutIndex == 0) {
        Module::draw(view, p, pass);
    } else {
        QImage tmp(aR.width(), aR.height(), QImage::Format_RGB32);
        for (int j = 0; j < tmp.height(); ++j) {
            for (int i = 0; i < tmp.width(); ++i) {
                QColor c = QColor::fromRgb(view->image().pixel(aR.x() + i, aR.y() + j)).toHsl();
                tmp.setPixel(i, j, QColor::fromHsl(
                    c.hslHue(),
                    0,
                    qBound(0, c.lightness()+64, 255),
                    255).rgb());
            }
        }
        p.drawImage(aR.x(), aR.y(), tmp);
        {
            p.setPen(QPen(Qt::gray, view->pt2px(0)));
            for (int i = aR.left(); i <= aR.right(); ++i) {
                QPointF q0(i, aR.top());
                QPointF q1(i, aR.bottom()+1);
                p.drawLine(q0,q1);
            }
            for (int i = aR.top(); i <=  aR.bottom(); ++i) {
                QPointF q0(aR.left(), i);
                QPointF q1(aR.right()+1, i);
                p.drawLine(q0,q1);
            }
        }
    }
    if (draw_st && (view->zoom() >= 2) && pass) {
        p.save();
        p.setPen(QPen(Qt::blue, 0.5*px, Qt::SolidLine, Qt::FlatCap));
        cpu_image st = publishedImage((layoutIndex != 2)? "st" : "st2");
        draw_minor_eigenvector_field(p, st, aR);
        p.restore();
    }
    if (draw_gf  && (view->zoom() >= 4) && pass) {
        p.save();

        QPen bPen(Qt::black, 0.33*px, Qt::SolidLine, Qt::FlatCap);
        QPen rPen(Qt::red, 0.33*px, Qt::SolidLine, Qt::FlatCap);

        cpu_image gf = publishedImage("gf");
        if (gf.is_valid()) {
            QPointF C = R.center();
            for (int j = aR.top(); j <= aR.bottom(); ++j) {
                for (int i = aR.left(); i <= aR.right(); ++i) {
                    if ((C - QPointF(i+0.5f,j+0.5f)).manhattanLength() <= 1.5f) {
                        p.setPen(bPen);
                    } else {
                        p.setPen(rPen);
                    }
                    float2 g = gf.at<float2>(i, j);
                    QPointF q(i+0.5, j+0.5);
                    QPointF v(0.5f * g.x, 0.5f * g.y);
                    draw_arrow(p, q, q+v);
                }
            }
        }
        p.restore();
    }
    if (draw_tf  && (view->zoom() >= 4) && pass) {
        p.save();

        QPen bPen(Qt::blue, 0.33*px, Qt::SolidLine, Qt::FlatCap);
        p.setPen(bPen);

        cpu_image gf = publishedImage("gf");
        if (gf.is_valid()) {
            QPointF C = R.center();
            for (int j = aR.top(); j <= aR.bottom(); ++j) {
                for (int i = aR.left(); i <= aR.right(); ++i) {
                    float2 g = gf.at<float2>(i, j);
                    QPointF q(i+0.5, j+0.5);
                    QPointF v(0.5f * g.y, -0.5f * g.x);
                    p.drawLine(q-v, q+v);
                }
            }
        }
        p.restore();
    }
    if ((draw_gradient || draw_tangent) && (!draw_mini || (pass == 1))) {
        p.save();
        QPointF q = QPointF(floor(R.center().x()) + 0.5f, floor(R.center().y()) + 0.5f);
        p.translate(q);

        //cpu_image st = publishedImage("st");
        //float3 s = st.at<float3>(q.x(), q.y());

        if (draw_angle) {
            double z = 0.33 * L;
            {
                QPainterPath path;
                path.moveTo(0, 0);
                path.arcTo(-z,-z, 2*z, 2*z, 0, -phi);
                path.closeSubpath();
                p.fillPath(path, Qt::white);
            }
            {
                QPainterPath path;
                path.moveTo(0, 0);
                path.arcTo(-z,-z, 2*z, 2*z, 0, -phi);
                path.closeSubpath();
                p.strokePath(path, QPen(Qt::black, 0.25*px));
            }
        }

        if (draw_axis) {
            p.setPen(QPen(Qt::black, 0.5*px, Qt::SolidLine, Qt::RoundCap));
            QPointF a(-0.5 * L, 0);
            QPointF b( 0.5 * L, 0);
            p.drawLine(a, b);
        }

        if (draw_tangent) {
            p.setPen(QPen(Qt::black, 0.5*px, Qt::DashLine, Qt::RoundCap));
            QPointF t(cos(radians(phi)), sin(radians(phi)));
            QPointF v = 0.5 * L * t;
            p.drawLine(-v, v);

            p.setPen(QPen(Qt::blue, 1*px, Qt::SolidLine, Qt::RoundCap));
            draw_arrow(p, QPointF(0,0), -0.33* L * t);
        }

        if (draw_gradient) {
            p.setPen(QPen(Qt::red, 1*px, Qt::SolidLine, Qt::RoundCap));
            QPointF g(sin(radians(phi)), -cos(radians(phi)));
            draw_arrow(p, QPointF(0,0), 0.33* L * g);
        }

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

        p.setPen(QPen(Qt::white, px, Qt::SolidLine, Qt::SquareCap, Qt::MiterJoin));
        int y = height - 6;
        QRectF L(0, y-px/2, width, 6);
        p.drawLine(L.topLeft(), L.topRight());
        //p.drawRect(R.adjusted(-px/4,-px/4,px/4,px/4));
        QRectF R(0, y, width, 6);
        p.drawImage(R, map, QRect(0,0,256,1));
    }
}

