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

    OZAPI gpu_image dog_filter( const gpu_image& src, float sigma_e, float sigma_r,
                                float tau0, float tau1, float precision = 2 );

    OZAPI gpu_image gradient_dog( const gpu_image& src, const gpu_image& tm,
                                  float sigma_e, float sigma_r, float tau0, float tau1,
                                  float precision=2 );

    OZAPI gpu_image dog_threshold_tanh( const gpu_image& src, float epsilon, float phi );
    OZAPI gpu_image dog_colorize( const gpu_image& src );
    OZAPI gpu_image dog_sign( const gpu_image& src );

    OZAPI gpu_image gradient_log( const gpu_image& src, const gpu_image& tm, float sigma, float tau );

}
