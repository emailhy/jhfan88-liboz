//
// TV-L1 optical flow:
// Antonin Chambolle and Thomas Pock, A first-order primal-dual
// algorithm with applications to imaging, Technical Report, 2010
//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// based on a MATLAB implementation by Thomas Pock 
// Copyright 2011 Adobe Systems Incorporated 
// 
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
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

    OZAPI gpu_image tvl1flow( const gpu_image& src0, const gpu_image& src1,
                              float pyr_scale, int warps, int maxits, float lambda );

}
