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

#include <oz/math_util.h>
#include <string>

namespace oz {

    enum image_format_t {
        FMT_INVALID,
        FMT_UCHAR,
        FMT_UCHAR2,
        FMT_UCHAR3,
        FMT_UCHAR4,
        FMT_FLOAT,
        FMT_FLOAT2,
        FMT_FLOAT3,
        FMT_FLOAT4,
        FMT_STRUCT
    };

    OZAPI unsigned image_format_type_size( image_format_t format );
    OZAPI unsigned image_format_channel( image_format_t format );
    OZAPI const char* image_format_str( image_format_t format );

    OZAPI std::string image_format_invalid_msg( image_format_t have, image_format_t want0,
                                                image_format_t want1=FMT_INVALID );

}
