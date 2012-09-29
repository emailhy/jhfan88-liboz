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

    struct deriv2nd_t {
        gpu_image Ix;
        gpu_image Iy;
        gpu_image Ixx;
        gpu_image Ixy;
        gpu_image Iyy;

        deriv2nd_t() {}
        deriv2nd_t( unsigned w, unsigned h, image_format_t f )
            : Ix(w,h,f), Iy(w,h,f), Ixx(w,h,f), Ixy(w,h,f), Iyy(w,h,f) {}
    };

    OZAPI deriv2nd_t deriv2nd( const gpu_image& src );

    OZAPI gpu_image deriv2nd_sign( const gpu_image& dir, const gpu_image& Ixx,
                                   const gpu_image& Ixy, const gpu_image& Iyy );

}
