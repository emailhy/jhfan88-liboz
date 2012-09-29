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
#include <oz/dir_gauss.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>


namespace {
    template<typename T> struct imp_dir_gauss : public oz::generator<T> {
        gpu_sampler<T,0> src_;
        oz::gpu_plm2<float2> tm_;
        float sigma_;
        float angle_;
        float precision_;

        imp_dir_gauss( const oz::gpu_image& src, const oz::gpu_image& tm,
                       float sigma, float angle, float precision )
            : src_(src), tm_(tm), sigma_(sigma), angle_(angle), precision_(precision) {}

        inline __device__ T operator()( int ix, int iy ) const {
            float twoSigma2 = 2 * sigma_ * sigma_;

            float2 n = rotate(tm_(ix, iy), angle_);
            float2 nabs = fabs(n);
            float ds = 1.0f / ((nabs.x > nabs.y)? nabs.x : nabs.y);

            T sum = src_(ix + 0.5f, iy + 0.5f);
            float norm = 1;

            float halfWidth = precision_ * sigma_;
            for( float d = ds; d <= halfWidth; d += ds ) {
                float k = __expf( -d * d / twoSigma2 );

                float2 o = d*n;
                T c = src_(ix + 0.5f + o.x, iy + 0.5f + o.y) +
                      src_(ix + 0.5f - o.x, iy + 0.5f - o.y);
                sum += k * c;
                norm += 2 * k;
            }
            sum /= norm;

            return sum;
        }
    };
}


oz::gpu_image oz::dir_gauss( const gpu_image& src, const gpu_image& tm,
                             float sigma, float angle, float precision ) {
    switch (src.format()) {
        case FMT_FLOAT:  return generate(src.size(), imp_dir_gauss<float >(src, tm, sigma, radians(angle), precision));
        case FMT_FLOAT3: return generate(src.size(), imp_dir_gauss<float3>(src, tm, sigma, radians(angle), precision));
        default:
            OZ_INVALID_FORMAT();
    }
}
