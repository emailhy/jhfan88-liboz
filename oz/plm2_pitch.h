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

#include <oz/type_traits.h>
#include <oz/gpu_image.h>

namespace oz {

    template <typename T> struct plm2_pitch {
        char *ptr;
        unsigned pitch;
        unsigned w;
        unsigned h;

        __host__ plm2_pitch() {
            ptr = 0;
            pitch = w = h = 0;
        }

        __host__ plm2_pitch( void *ptr, unsigned pitch, unsigned w, unsigned h ) {
            this->ptr = static_cast<char*>(ptr);
            this->pitch = pitch;
            this->w = w;
            this->h = h;
        }

        __host__ plm2_pitch( const gpu_image& img ) {
            OZ_CHECK_FORMAT(img.format(), type_traits<T>::format());
            ptr = img.ptr<char>();
            pitch = img.pitch();
            w = img.w();
            h = img.h();
        }

        inline __host__ __device__ T read( int x, int y ) const {
            return *( (T*)( ((char*)ptr) + y * pitch + sizeof(T)*x) );
        }

        inline __host__ __device__ void write( int x, int y, T value ) {
            *( (T*)( ((char*)ptr) + y * pitch + sizeof(T)*x) ) = value;
        }

        inline __host__ __device__ const T operator()( int x, int y ) const {
            return read(x, y);
        }
    };

}


