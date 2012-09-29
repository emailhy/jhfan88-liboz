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

    OZAPI gpu_image grad_gaussian( const gpu_image& src, float sigma, float precision, bool normalize=false );
    OZAPI gpu_image grad_central_diff( const gpu_image& src, bool normalize=false );
    OZAPI gpu_image grad_sobel( const gpu_image& src, bool normalize=false );
    OZAPI gpu_image grad_scharr_3x3( const gpu_image& src, bool normalize=false );
    OZAPI gpu_image grad_scharr_5x5( const gpu_image& src, bool normalize=false );

    OZAPI gpu_image grad_to_axis( const gpu_image& src, bool squared );
    OZAPI gpu_image grad_from_axis( const gpu_image& src, bool squared );
    OZAPI gpu_image grad_angle( const gpu_image& src, bool perpendicular=false );
    OZAPI gpu_image grad_to_lfm( const gpu_image& src );

    OZAPI gpu_image grad_sobel_mag( const gpu_image& src );

}
