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
#include "apparentgray.h"
#include <oz/color.h>
#include <oz/pyr.h>
//#include <oz/colormap.h>
#include <oz/make.h>
#include <oz/shuffle.h>
#include <oz/apparent_gray.h>
#include <oz/hist.h>
#include <oz/qimage.h>
using namespace oz;


QImage renderhPyr2(const std::vector<gpu_image>& P, int orientation=-1, bool drawScale=false) {
    if (P.size() == 0 ) {
        return QImage();
    }
    if (P.size() == 1 ) {
        return to_qimage(P[0]);
    }
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

    /*if (drawScale) {
        QImage map(256,1,QImage::Format_RGB32);
        for (int i = 0; i < 256; ++i) {
            float4 c = colormap_jet(1.0f * i / 255.0f);
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
    }*/

    return img;
}


ApparentGray::ApparentGray() {
    new ParamInt   (this, "N", 4, 1, 100, 1, &N);
    new ParamDouble(this, "pi", 0.5, 0, 1, 0.01, &pi);
    new ParamDouble(this, "ki", 0.5, 0, 10, 0.01, &ki);
}


void ApparentGray::process() {
    gpu_image src = gpuInput0();
    gpu_image lab = rgb2lab(src);
    gpu_image Lnvac = rgb2nvac(src);
    lab = make(lab, Lnvac);
    publish("$src", src);
    publish("Lnvac", Lnvac/100.f);

    std::vector<gpu_image> LP;
    gpu_image img = lab;
    for (int k = 0; k < N; ++k) {
        gpu_image dw = pyrdown_gauss5x5(img);
        gpu_image up = pyrup_gauss5x5(dw, img.w(), img.h());
        gpu_image hp = img - up;
        LP.push_back(hp);
        img = dw;
    }
    LP.push_back(img);

    std::vector<gpu_image> W;
    for (int k = N-1; k >= 0; --k) {
        publish(QString("img-%1").arg(k), shuffle(img,3)/100.0f);
        gpu_image up = pyrup_gauss5x5(img, LP[k].w(), LP[k].h());
        W.push_back(apparent_gray_weight(up, LP[k], ki, pi));
        img = apparent_gray_sharpen(up, LP[k], ki, pi);
    }

    std::vector<gpu_image> W2;
    for (int i = (int)W.size()-1; i >=0 ; --i) W2.push_back(W[i]);
    publish("W2", renderhPyr2(W2, 0));

    for (int i = 0; i < LP.size(); ++i) LP[i] = lab2rgb(LP[i].convert(FMT_FLOAT3)) * ((i != LP.size()-1)? 10.0f : 1.0f);
    publish("LP", renderhPyr2(LP, 0));
    publish("R", lab2rgb(img.convert(FMT_FLOAT3)));
    publish("AG", shuffle(img, 3)/100.0f);
    publish("L", shuffle(lab, 3)/100.0f);

    std::vector<float> k(N, ki);
    publish("AG2", hist_auto_levels(apparent_gray(src, N, &k[0], pi)/100.0f));
}

