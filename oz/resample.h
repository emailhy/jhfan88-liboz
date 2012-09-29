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

    enum resample_mode_t {
        RESAMPLE_BOX,
        RESAMPLE_TRIANGLE,
        RESAMPLE_BELL,
        RESAMPLE_QUADRATIC,
        RESAMPLE_QUADRATIC_APPROX,
        RESAMPLE_QUADRATIC_MIX,
        RESAMPLE_BSPLINE,
        RESAMPLE_LANCZOS2,
        RESAMPLE_LANCZOS3,
        RESAMPLE_BLACKMAN,
        RESAMPLE_CUBIC,
        RESAMPLE_CATROM,
        RESAMPLE_MITCHELL,
        RESAMPLE_GAUSSIAN,
        RESAMPLE_KAISER
    };

    OZAPI gpu_image resample( const gpu_image& src, unsigned w, unsigned h, resample_mode_t mode );
    OZAPI gpu_image resample_gaussian( const gpu_image& src, unsigned w, unsigned h, float sigma, float precision=3 );
    OZAPI gpu_image resample_boxed( const gpu_image& src, unsigned max_w, unsigned max_h );

}
