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

#include <oz/config.h>

namespace oz {

    class cpu_image;
    class gpu_image;

    OZAPI void log_debug( const char* format, ... );
    OZAPI void log_warn( const char* format, ... );
    OZAPI void log_stack();
    OZAPI void log_image( const cpu_image& image, const char* format, ... );
    OZAPI void log_image( const gpu_image& image, const char* format, ... );
    OZAPI void install_log_handler(void (*callback)(const cpu_image&, const char*, void*), void *user);

}
