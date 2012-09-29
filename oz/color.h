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

    OZAPI gpu_image srgb2linear( const gpu_image& src );
    OZAPI gpu_image linear2srgb( const gpu_image& src );

    OZAPI gpu_image gray2rgb( const gpu_image& src );
    OZAPI gpu_image rgb2gray( const gpu_image& src );

    OZAPI gpu_image rgb2lab( const gpu_image& src );
    OZAPI gpu_image lab2rgb( const gpu_image& src );

    OZAPI gpu_image rgb2luv( const gpu_image& src );
    OZAPI gpu_image luv2rgb( const gpu_image& src );
    OZAPI gpu_image rgb2nvac( const gpu_image& src );

    OZAPI gpu_image swap_rgb( const gpu_image& src );

}