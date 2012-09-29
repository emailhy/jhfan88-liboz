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

#include <cstddef>
#include <oz/config.h>

namespace oz {

    OZAPI void gpu_cache_alloc( void **ptr, unsigned *pitch, unsigned w, unsigned h );
    OZAPI void gpu_cache_free( void *ptr, unsigned pitch, unsigned w, unsigned h );
    OZAPI void gpu_cache_clear();
    OZAPI size_t gpu_cache_size();
    OZAPI size_t gpu_cache_total();

}
