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
#include <oz/conv.h>
#include <oz/generate.h>
#include <oz/gpu_sampler2.h>


namespace oz {

    template<typename T> struct Conv : public generator<T> {
        gpu_sampler<float,0> krnl_;
        gpu_sampler<T,1> src_;

        Conv( const gpu_image& krnl, const gpu_image& src )
            : krnl_(krnl), src_(src) {}

        inline __device__ T operator()( int ix, int iy) const {
            int kw2 = krnl_.w / 2;
            int kh2 = krnl_.h / 2;
            T sum = make_zero<T>();
            float sum_w = 0;
            for (int j = 0; j < krnl_.h; ++j) {
                for (int i = 0; i < krnl_.w; ++i) {
                    float k = krnl_(i, j);
                    T c = src_(ix - kw2 + i, iy - kh2 + j);
                    sum += k * c;
                    sum_w += k;
                }
            }
            return sum / sum_w;
        }
    };

    gpu_image conv( const gpu_image& krnl, const gpu_image& src ) {
        switch (src.format()) {
            case FMT_FLOAT:  return generate(src.size(), Conv<float >(krnl, src));
            case FMT_FLOAT3: return generate(src.size(), Conv<float3>(krnl, src));
            default:
                OZ_INVALID_FORMAT();
        }
    }

}
