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
#include <oz/road.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>


namespace {
    template<typename T> struct Road3x3 : public oz::generator<float> {
        gpu_sampler<T,0> src_;

        Road3x3( const oz::gpu_image& src ) : src_(src) {}

        inline __device__ float operator()( int ix, int iy) const {
            T c0 = src_(ix, iy);
            float tmp[8];
            tmp[0] = length(src_(ix - 1, iy + 1) - c0);
            tmp[1] = length(src_(ix    , iy + 1) - c0);
            tmp[2] = length(src_(ix + 1, iy + 1) - c0);
            tmp[3] = length(src_(ix - 1, iy    ) - c0);
            tmp[4] = length(src_(ix + 1, iy    ) - c0);
            tmp[5] = length(src_(ix - 1, iy - 1) - c0);
            tmp[6] = length(src_(ix    , iy - 1) - c0);
            tmp[7] = length(src_(ix + 1, iy - 1) - c0);

            const int N = 8;
            for (int i = 0; i < N; i++) {
                for (int j = N-1; j > i; j--) {
                    if (tmp[j-1] > tmp[j]) {
                        float t = tmp[j-1];
                        tmp[j-1] = tmp[j];
                        tmp[j] = t;
                    }
                }
            }

            return (tmp[0] + tmp[1] + tmp[2] + tmp[3]) / 4;
        }
    };
}


oz::gpu_image oz::road4_3x3( const gpu_image& src ) {
    switch (src.format()) {
        case FMT_FLOAT:  return generate(src.size(), Road3x3<float >(src));
        case FMT_FLOAT3: return generate(src.size(), Road3x3<float3>(src));
        default:
            OZ_INVALID_FORMAT();
    }
}
