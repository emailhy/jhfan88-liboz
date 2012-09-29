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
#include <cuComplex.h>

namespace oz {

    OZAPI gpu_image fftshift( const gpu_image& src );
    OZAPI gpu_image fft2( const gpu_image& src );

    OZAPI gpu_image psf_padshift( unsigned w, unsigned h, const gpu_image& psf );

    OZAPI void fft_complex_norm_mul( cuFloatComplex *c, const cuFloatComplex *a,
                                     const cuFloatComplex *b, float scale, unsigned N );

    OZAPI void fft_pad_wrap( float *dst, unsigned w, unsigned h, const gpu_image& src );
    OZAPI void fft_pad_shift( float *dst, unsigned w, unsigned h, const gpu_image& src );

}
