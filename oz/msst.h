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
#include <oz/resample.h>
#include <oz/st.h>

namespace oz {

    OZAPI gpu_image st_moa_merge( const gpu_image& stA, const gpu_image& stB,
                                  float epsilon, moa_t moa );

    OZAPI gpu_image st_multi_scale( const gpu_image& src, int max_depth, float rho,
                                    resample_mode_t resample_mode, float epsilon, moa_t moa );

}
