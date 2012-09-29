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
#include <vector>

namespace oz {

    OZAPI gpu_image oabf_1d( const gpu_image& src, const gpu_image& lfm, float sigma_d, float sigma_r,
                             bool tangential, bool src_linear, bool ustep, float precision );

    OZAPI gpu_image oabf( const gpu_image& src, const gpu_image& lfm, float sigma_d, float sigma_r,
                          bool src_linear, bool ustep, float precision);

    OZAPI std::vector<float3> oabf_line( int ix, int iy, const cpu_image& lfm, float sigma_d,
                                         float sigma_r, bool tangential, float precision );

    OZAPI int oabf_sample_dir( int ix, int iy, const cpu_image& lfm, bool tangential );
}

