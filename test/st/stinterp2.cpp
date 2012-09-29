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
#include "stinterp2.h"
#include <oz/st.h>
#include <oz/resize.h>
#include <oz/qpainter_draw.h>
using namespace oz;


StInterp2::StInterp2() {
    new ParamDouble(this, "rho", 2.0, 0.0, 10.0, 0.1, &rho);
    new ParamInt   (this, "nx", 1, 1, 100, 1, &nx);
    new ParamInt   (this, "ny", 1, 1, 100, 1, &ny);
    new ParamBool  (this, "bilinear", true, &bilinear);
}


void StInterp2::process() {
    gpu_image src = gpuInput0();
    publish("src", src);
    gpu_image st = st_scharr_3x3(src, rho);
    publish("st", st);
    gpu_image st2 = resize(st, src.w() * nx, src.h() * ny, bilinear? RESIZE_BILINEAR : RESIZE_NEAREST);
    publish("st2", st2);
}


void StInterp2::draw(ImageView *view, QPainter &p, int pass) {
    Module::draw(view, p, pass);

    QRectF R = p.clipBoundingRect();
    QRect aR = R.toAlignedRect().intersected(view->image().rect());
    double pt = view->pt2px(1);

    if (view->zoom() > 10) {
        p.save();
        p.scale(1.0/nx, 1.0/ny);
        p.setPen(QPen(Qt::red, nx*0.25*pt));
        cpu_image st2 = publishedImage("st2");
        draw_minor_eigenvector_field(p, st2, QRect(aR.x()*nx, aR.y()*ny, aR.width()*nx, aR.height()*ny));
        p.restore();

        p.setPen(QPen(Qt::blue, 0.5*pt));
        cpu_image st = publishedImage("st");
        draw_minor_eigenvector_field(p, st, aR);
    }
}
