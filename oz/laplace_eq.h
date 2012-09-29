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

    enum leq_stencil_t {
        LEQ_STENCIL_4 = 0,
        LEQ_STENCIL_8,
        LEQ_STENCIL_12,
        LEQ_STENCIL_20
    };

    enum leq_upfilt_t {
        LEQ_UPFILT_NEAREST = 0,
        LEQ_UPFILT_FAST_BILINEAR,
        LEQ_UPFILT_BILINEAR,
        LEQ_UPFILT_FAST_BICUBIC,
        LEQ_UPFILT_BICUBIC
    };

    OZAPI gpu_image leq_jacobi_step( const gpu_image& src, leq_stencil_t );
    OZAPI gpu_image leq_correct_down( const gpu_image& src );
    OZAPI gpu_image leq_correct_up( const gpu_image& src0, const gpu_image& src1, leq_upfilt_t upfilt );
    OZAPI gpu_image leq_residual( const gpu_image& src, leq_stencil_t );
    OZAPI float leq_error( const gpu_image& src, leq_stencil_t stencil );
    OZAPI gpu_image leq_vcycle( const gpu_image& b, int v2, leq_stencil_t, leq_upfilt_t upfilt );

}
