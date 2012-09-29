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
#include <oz/temp_gauss.h>
#include <oz/generate.h>
#include <oz/gpu_sampler5.h>


namespace oz {
    template<typename T> struct imp_temp_gauss_5 : public generator<T> {
        gpu_sampler<T,0> src0_;
        gpu_sampler<T,1> src1_;
        gpu_sampler<T,2> src2_;
        gpu_sampler<T,3> src3_;
        gpu_sampler<T,4> src4_;

        imp_temp_gauss_5( const gpu_image& src0, const gpu_image& src1, const gpu_image& src2,
                          const gpu_image& src3, const gpu_image& src4)
            : src0_(src0), src1_(src1), src2_(src2), src3_(src3), src4_(src4) {}

        inline __device__ T operator()( int ix, int iy ) const {
            T c0 = src0_(ix, iy);
            T c1 = src1_(ix, iy);
            T c2 = src2_(ix, iy);
            T c3 = src3_(ix, iy);
            T c4 = src4_(ix, iy);
            return ( 1*c0 + 4*c1 + 6*c2 + 4*c3 + 1*c4 ) / 16.0;
        }
    };
}


oz::gpu_image oz::temp_gauss_5( const gpu_image& src0,
                                const gpu_image& src1,
                                const gpu_image& src2,
                                const gpu_image& src3,
                                const gpu_image& src4 )
{
    if ((src0.size() != src1.size()) ||
        (src0.size() != src2.size()) ||
        (src0.size() != src3.size()) ||
        (src0.size() != src4.size())) OZ_INVALID_SIZE();

    switch (src0.format()) {
        case FMT_FLOAT3: return generate(src0.size(), imp_temp_gauss_5<float>(src0, src1, src2, src3, src4));
        default:
            OZ_INVALID_FORMAT();
    }
}
