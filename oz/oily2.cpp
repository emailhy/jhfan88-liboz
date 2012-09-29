//
// by Jan Eric Kyprianidis and Daniel Müller
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
#include <oz/oily2.h>
#include <oz/stgauss.h>
#include <oz/bump.h>
#include <oz/blend.h>


oz::gpu_image oz::oily2( const gpu_image& src, const gpu_image& st, const gpu_image& noise,
                        float flow_blur, float bump_scale,
                        float phong_specular, float phong_shininess)
{
    gpu_image I = stgauss_filter(noise, st, flow_blur, 90.0f, false);
    I = bump_phong(I, bump_scale, phong_specular, phong_shininess);

    gpu_image dst = stgauss_filter(src, st, flow_blur, 90.0f, false);
    return blend_intensity(dst, I, BLEND_LINEAR_DODGE);
}
