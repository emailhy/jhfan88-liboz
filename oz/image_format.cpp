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
#include <oz/image_format.h>
#include <oz/type_traits.h>
#include <sstream>


unsigned oz::image_format_type_size( image_format_t format ) {
    static unsigned s_format_size[] = {
        0,
        sizeof(type_traits<unsigned char>::texture_type),
        sizeof(type_traits<uchar2>::texture_type),
        sizeof(type_traits<uchar3>::texture_type),
        sizeof(type_traits<uchar4>::texture_type),
        sizeof(type_traits<float>::texture_type),
        sizeof(type_traits<float2>::texture_type),
        sizeof(type_traits<float3>::texture_type),
        sizeof(type_traits<float4>::texture_type),
        0
    };
    return s_format_size[format];
};


unsigned oz::image_format_channel( image_format_t format ) {
    static unsigned s_format_channel[] = {
        0,
        sizeof(type_traits<unsigned char>::N),
        sizeof(type_traits<uchar2>::N),
        sizeof(type_traits<uchar3>::N),
        sizeof(type_traits<uchar4>::N),
        sizeof(type_traits<float>::N),
        sizeof(type_traits<float2>::N),
        sizeof(type_traits<float3>::N),
        sizeof(type_traits<float4>::N),
        0
    };
    return s_format_channel[format];
};


const char* oz::image_format_str( image_format_t format ) {
    static const char* s_format_str[] = {
        "*invalid*",
        "uchar", "uchar2", "uchar3", "uchar4",
        "float", "float2", "float3", "float4",
        "*struct*"
    };
    return s_format_str[format];
};


std::string oz::image_format_invalid_msg( image_format_t have, image_format_t want0, image_format_t want1 ) {
    std::stringstream s;
    if (want1 == FMT_INVALID) {
        s << "Invalid format: "
          << image_format_str(have)
          << " != " \
          << image_format_str(want0);
    } else {
        s << "Invalid format: " \
          << image_format_str(have) \
          << " != " \
          << image_format_str(want0) << "|" << image_format_str(want1);
    }
    return s.str();
}
