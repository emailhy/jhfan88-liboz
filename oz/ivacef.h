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

    OZAPI gpu_image ivacef_sobel( const gpu_image& src, const gpu_image& prev, float tau_r );

    OZAPI gpu_image ivacef_relax( const gpu_image& st, int v2 );

    OZAPI gpu_image ivacef_compute_st( const gpu_image& src, const gpu_image& prev,
                                       float sigma_d, float tau_r, int v2 );

    OZAPI gpu_image ivacef_sign( const gpu_image& L, const gpu_image& st, float sigma, float tau );

    OZAPI gpu_image ivacef_shock( const gpu_image& src, const gpu_image& st,
                                  const gpu_image& sign, float radius );

    OZAPI gpu_image ivacef( const gpu_image& src, int N=5,
                            float sigma_d=1, float tau_r=0.002f, int v2=1, float sigma_t=6,
                            float max_angle=22.5f, float sigma_i=0, float sigma_g=1.5f,
                            float r=2, float tau_s=0.005f, float sigma_a=1.5f, bool adaptive=true );

}
