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
#include <oz/qpainter_draw.h>
#include <oz/st_util.h>


double oz::draw_pt2px(QPainter &p) {
    QLineF L = p.deviceTransform().inverted().map(QLineF(0, 0, 1, 1));
    return qMax(L.dx(), L.dy());
}


void oz::draw_points(QPainter &p, const QPolygonF polygon, float radius, const QBrush& brush) {
    for (int i= 0; i < polygon.size(); ++i) {
        QPainterPath path;
        path.addEllipse(polygon[i], radius, radius);
        p.fillPath(path, brush);
    }
}


void oz::draw_arrow(QPainter &p, const QPointF& a, const QPointF& b, float size) {
    float pt = p.pen().widthF() * size;

    QPointF v = b - a;
    float len = sqrtf(v.x()*v.x() + v.y()*v.y());
    v /= len;
    QPointF w(-v.y(), v.x());

    QPointF q = a + (len - pt) * v;
    p.drawLine(a, q);

    QPainterPath P;
    P.moveTo(b);
    P.lineTo(b - 2*pt * v + pt * w);
    P.lineTo(b - 2*pt * v - pt * w);
    P.closeSubpath();
    p.fillPath(P, p.pen().color());
}


void oz::draw_vector_field(QPainter &p, const cpu_image& vf, const QRect& R, bool arrows) {
    if (!vf.is_valid()) return;

    for (int j = R.top(); j <= R.bottom(); ++j) {
        for (int i = R.left(); i <= R.right(); ++i) {
            float2 t = vf.at<float2>(i, j);
            QPointF q(i+0.5, j+0.5);
            QPointF v(0.5f * t.x, 0.5f * t.y);
            if (arrows) {
                draw_arrow(p, q, q+v);
            } else {
                p.drawLine(q-v, q+v);
            }
        }
    }
}


void oz::draw_orientation_field(QPainter &p, const cpu_image& vf, const QRect& R) {
    if (!vf.is_valid()) return;

    for (int j = R.top(); j <= R.bottom(); ++j) {
        for (int i = R.left(); i <= R.right(); ++i) {
            float2 t = normalize(vf.at<float2>(i, j));
            QPointF q(i+0.5, j+0.5);
            QPointF v(-0.5f * t.y, 0.5f * t.x);
            p.drawLine(q-v, q+v);
        }
    }
}


void oz::draw_minor_eigenvector_field(QPainter &p, const cpu_image& st, const QRect& R) {
    if (!st.is_valid()) return;

    for (int j = R.top(); j <= R.bottom(); ++j) {
        for (int i = R.left(); i <= R.right(); ++i) {
            float3 g = st.at<float3>(i, j);
            float2 t = st2tangent(g);
            QPointF q(i+0.5, j+0.5);
            QPointF v(0.4 * t.x, 0.4 * t.y);
            p.drawLine(q-v, q+v);
        }
    }
}
