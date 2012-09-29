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
#include <oz/gpu_image.h>
#include <cairo.h>

namespace oz {

    OZAPI cairo_surface_t* create_surface( const cpu_image& src );
    OZAPI void draw_image( cairo_t *cr, const cpu_image& src, double x=0, double y=0 );
    OZAPI void draw_image( cairo_t *cr, const gpu_image& src, double x=0, double y=0 );

}
