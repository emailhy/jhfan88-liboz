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

    template <typename T> struct gpu_plm2 {
        typedef typename type_traits<T>::texture_type TT;
        TT *ptr;
        unsigned stride;
        unsigned w;
        unsigned h;

        __host__ gpu_plm2() {
            ptr = 0;
            stride = w = h = 0;
        }

        __host__ gpu_plm2( void *ptr, unsigned stride, unsigned w, unsigned h ) {
            this->ptr = static_cast<TT*>(ptr);
            this->stride = stride;
            this->w = w;
            this->h = h;
        }

        __host__ gpu_plm2( const gpu_image& img ) {
            OZ_CHECK_FORMAT(img.format(), type_traits<T>::format());
            ptr = img.ptr<TT>();
            stride = img.stride();
            w = img.w();
            h = img.h();
        }

        inline __host__ __device__ T read( int i ) const {
            return make_T<T>(ptr[i]);
        }

        inline __host__ __device__ T read( int x, int y ) const {
            return read(y * stride + x);
        }

        inline __host__ __device__ void write( int i, T value ) {
            ptr[i] = make_T<TT>(value);
        }

        inline __host__ __device__ void write( int x, int y, T value ) {
            write(y * stride + x, value);
        }

        inline __host__ __device__ const T operator()( int x, int y ) const {
            return read(x, y);
        }
    };

}


