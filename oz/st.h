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

    enum moa_t {
        MOA_SQUARED_EV = 0,
        MOA_SQRT_EV,
        MOA_SQUARED_EV2
    };

    OZAPI gpu_image st_central_diff( const gpu_image& src );
    OZAPI gpu_image st_gaussian( const gpu_image& src, float rho, float precision, bool normalize=false );
    OZAPI gpu_image st_gaussian_x2( const gpu_image& src, float rho, float precision, bool normalize=false );
    OZAPI gpu_image st_sobel( const gpu_image& src, float rho=0 );
    OZAPI gpu_image st_scharr_3x3( const gpu_image& src, float rho=0, bool normalize=false );
    OZAPI gpu_image st_scharr_5x5( const gpu_image& src, float rho=0, bool normalize=false );
    OZAPI gpu_image st_from_gradient( const gpu_image& g );
    OZAPI gpu_image st_from_tangent( const gpu_image& t );

    OZAPI gpu_image st_to_tangent( const gpu_image& st );
    OZAPI gpu_image st_to_gradient( const gpu_image& st );
    OZAPI gpu_image st_lfm( const gpu_image& st, float alpha=0 );
    OZAPI gpu_image st_lfm2( const gpu_image& st, moa_t moa, float alpha=0 );
    OZAPI gpu_image st_angle( const gpu_image& st );
    OZAPI gpu_image st_moa( const gpu_image& st, moa_t moa );

    //OZAPI gpu_image st_threshold_mag( const gpu_image& st, float threshold );
    OZAPI gpu_image st_normalize( const gpu_image& st );
    OZAPI gpu_image st_flatten( const gpu_image& st );
    OZAPI gpu_image st_rotate( const gpu_image& st, float angle );
    OZAPI gpu_image st_pow( const gpu_image& st, float y );
    OZAPI gpu_image st_exp( const gpu_image& st );
    OZAPI gpu_image st_log( const gpu_image& st );

}
