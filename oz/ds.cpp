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
#include <oz/ds.h>
#include <oz/grad.h>
#include <oz/st.h>
#include <oz/gauss.h>


oz::gpu_image oz::ds_scharr_3x3( const gpu_image& src, float rho, bool normalize, bool squared ) {
    if (src.format() != FMT_FLOAT) OZ_INVALID_FORMAT();
    gpu_image g = grad_scharr_3x3(src, normalize);
    g = grad_to_axis(g, squared);
    g = gauss_filter_xy(g, rho);
    g = grad_from_axis(g, squared);
    return st_from_gradient(g);
}
