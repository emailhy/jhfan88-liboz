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

#include <oz/gpu_image.h>

namespace oz {

    enum resize_mode_t {
        RESIZE_NEAREST,
        RESIZE_FAST_BILINEAR,
        RESIZE_BILINEAR,
        RESIZE_FAST_BICUBIC,
        RESIZE_BICUBIC,
        RESIZE_CATROM
    };

    OZAPI gpu_image resize( const gpu_image& src, unsigned w, unsigned h, resize_mode_t mode );
    OZAPI gpu_image resize_x2( const gpu_image& src );
    OZAPI gpu_image resize_half( const gpu_image& src );

}