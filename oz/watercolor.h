//
// by Daniel Müller and Jan Eric Kyprianidis
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

    OZAPI gpu_image watercolor( const gpu_image& src,
                                const gpu_image& st,
                                const gpu_image& noise,
                                float bf_sigma_d = 4,
                                float bf_sigma_r = 15,
                                int bf_N = 4,
                                int nbins = 16,
                                float phi_q = 2,
                                float sigma_c = 4,
                                float nalpha = 0.1,
                                float nbeta = 0.4 );

}
