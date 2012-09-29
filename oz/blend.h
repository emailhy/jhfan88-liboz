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
#pragma once

#include <oz/gpu_image.h>

namespace oz {

    enum blend_mode_t {
        BLEND_NORMAL,
        BLEND_MULTIPLY,
        BLEND_SCREEN,
        BLEND_HARD_LIGHT,
        BLEND_SOFT_LIGHT,
        BLEND_OVERLAY,
        BLEND_LINEAR_BURN,
        BLEND_DIFFERENCE,
        BLEND_LINEAR_DODGE
    };

    OZAPI gpu_image blend( const gpu_image& back, const gpu_image& src, blend_mode_t mode );

    OZAPI gpu_image blend_intensity( const gpu_image& back, const gpu_image& src,
                                     blend_mode_t mode, float4 color = make_float4(1) );

}
