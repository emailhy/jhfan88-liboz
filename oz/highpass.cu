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
#include <oz/highpass.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>


namespace {
    template<typename T> struct HighPass : public oz::generator<T> {
        gpu_sampler<T,0> src_;
        int radius_;

        HighPass( const oz::gpu_image& src, int radius) : src_(src), radius_(radius) {}

        inline __device__ T operator()( int ix, int iy ) const {
            T sum = make_zero<T>();
            for (int j = -radius_; j <= radius_; ++j) {
                for (int i = -radius_; i <= radius_; ++i) {
                    sum += src_(ix + i, iy + j);
                }
            }
            int d = 2 * radius_ + 1;
            return src_(ix, iy) - sum / (d*d);
        }
    };
}


oz::gpu_image oz::highpass( const gpu_image& src, int radius ) {
    switch (src.format()) {
        case FMT_FLOAT:  return generate(src.size(), HighPass<float >(src, radius));
        case FMT_FLOAT3: return generate(src.size(), HighPass<float3>(src, radius));
        default:
            OZ_INVALID_FORMAT();
    }
}
