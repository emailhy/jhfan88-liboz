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

    OZAPI gpu_image noise_random( unsigned w, unsigned h, float a=0, float b=1 );
    OZAPI gpu_image noise_fast( unsigned w, unsigned h, float scale );
    OZAPI gpu_image noise_fast2( unsigned w, unsigned h, float scale, float micro );

    OZAPI gpu_image noise_uniform( unsigned w, unsigned h, float a=0, float b=1 );
    OZAPI gpu_image noise_normal( unsigned w, unsigned h, float mean=0, float variance=0.01f );

    OZAPI gpu_image add_gaussian_noise( const gpu_image& src, float mean=0, float variance=0.01f );
    OZAPI gpu_image add_salt_and_pepper_noise( const gpu_image& src, float density=0.05f );
    OZAPI gpu_image add_speckle_noise( const gpu_image& src, float variance=0.04f );

}
