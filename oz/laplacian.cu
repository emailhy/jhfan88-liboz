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
#include <oz/laplacian.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>


namespace oz {
    template <typename T> struct Laplacian : public oz::generator<T> {
        gpu_sampler<T,0> src_;

        Laplacian( const oz::gpu_image& src ) : src_(src) {}

        inline __device__ T operator()( int ix, int iy ) const {
            return src_(ix-1, iy) +
                   src_(ix+1, iy) +
                   src_(ix, iy-1) +
                   src_(ix, iy+1) - 4 * src_(ix, iy);
        }
    };


    gpu_image laplacian( const gpu_image& src ) {
        switch (src.format()) {
            case FMT_FLOAT:  return generate(src.size(), Laplacian<float>(src));
            case FMT_FLOAT3: return generate(src.size(), Laplacian<float3>(src));
            default:
                OZ_INVALID_FORMAT();
        }
    }
}


