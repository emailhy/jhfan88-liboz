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

    OZAPI gpu_image test_circle( int width, int height, float r );
    OZAPI gpu_image test_wiggle( int width, int height, float r );
    OZAPI gpu_image test_line( int width, int height, float phi, float r );
    OZAPI gpu_image test_simple( int width, int height, float phi, float phase, float scale, int function );
    OZAPI gpu_image test_sphere( int width, int height );
    OZAPI gpu_image test_grad3( int width, int height );

    OZAPI gpu_image test_knutsson_ring();
    OZAPI gpu_image test_jaenich_ring( int width, int height, float g0, float km, float rm, float w );
    OZAPI gpu_image test_zoneplate( int width, int height, float g0, float km, float rm, float w, bool inverted=false );

    OZAPI gpu_image test_color_fan( int width, int height );

}
