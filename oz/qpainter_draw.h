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
#pragma once

#include <oz/cpu_image.h>
#include <qpainter.h>

namespace oz {

    OZAPI double draw_pt2px(QPainter &p);
    OZAPI void draw_points(QPainter &p, const QPolygonF polygon, float radius, const QBrush& brush);
    OZAPI void draw_arrow(QPainter &p, const QPointF& a, const QPointF& b, float size=2);
    OZAPI void draw_vector_field(QPainter &p, const cpu_image& vf, const QRect& R, bool arrows);
    OZAPI void draw_orientation_field(QPainter &p, const cpu_image& vf, const QRect& R);
    OZAPI void draw_minor_eigenvector_field(QPainter &p, const cpu_image& st, const QRect& R);

}
