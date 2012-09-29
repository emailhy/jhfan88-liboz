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
#include <oz/cairo_support.h>


static const cairo_user_data_key_t g_surface_key = { 0 };


namespace oz {

    struct cairo_surface_data {
        cairo_surface_data( const cpu_image& image) : image_(image) {}
        cpu_image image_;

        static void destory(void *data) {
            cairo_surface_data *self = (cairo_surface_data*)data;
            if (self) {
                delete self;
            }
        }
    };

}


cairo_surface_t* oz::create_surface( const cpu_image& src ) {
    cairo_format_t format;
    switch (src.format()) {
        case FMT_UCHAR:
            format = CAIRO_FORMAT_A8;
            break;

        case FMT_UCHAR3:
            format = CAIRO_FORMAT_RGB24;
            break;

        case FMT_UCHAR4:
            format = CAIRO_FORMAT_ARGB32;
            break;

        default:
            OZ_INVALID_FORMAT();
    }

    cairo_surface_t *surface = cairo_image_surface_create_for_data (
        (uchar*)src.ptr(),
        format,
        src.w(), src.h(), src.pitch());

    if (!surface)
        OZ_X() << "Error: Creation of cairo surface failed!";

    cairo_surface_data *user_data = new cairo_surface_data(src);
    cairo_surface_set_user_data(surface, &g_surface_key, user_data, cairo_surface_data::destory);

    return surface;
}


void oz::draw_image( cairo_t *cr, const cpu_image& src, double x, double y ) {
    cairo_save(cr);
    cairo_rectangle(cr, x, y, src.w(), src.h());
    cairo_clip(cr);
    double x1, y1, x2, y2;
    cairo_clip_extents(cr, &x1, &y1, &x2, &y2);
    cairo_restore(cr);
    if ((x1 == 0) && (y1 == 0) && (x2 == 0) && (y2 == 0)) {
        return;
    }

    int sx1 = (int)floor(x1 - x);
    int sy1 = (int)floor(y1 - y);
    int sx2 = (int)floor(x2 - x - 1e-5);
    int sy2 = (int)floor(y2 - y - 1e-5);

    cairo_save(cr);
    cpu_image src_copy = src.copy(sx1, sy1, sx2, sy2);
    cairo_surface_t *surface = create_surface(src_copy);
    cairo_set_source_surface(cr, surface, x + sx1, y + sy1);
    cairo_pattern_set_filter(cairo_get_source(cr), CAIRO_FILTER_NEAREST);
    cairo_paint(cr);
    cairo_surface_destroy(surface);
    cairo_restore(cr);
}


void oz::draw_image( cairo_t *cr, const gpu_image& src, double x, double y ) {
    draw_image(cr, src.cpu(), x, y);
}
