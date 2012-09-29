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
#include <oz/aniso_gauss.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>


namespace oz {
    template<typename T> struct imp_aniso_gauss : public generator<T> {
        gpu_sampler<T,0> src_;
        gpu_plm2<float4> lfm_;
        float sigma_;
        float precision_;

        imp_aniso_gauss( const gpu_image& src, const gpu_image& lfm, float sigma, float precision )
            : src_(src), lfm_(lfm), sigma_(sigma), precision_(precision) {}

        inline __device__ T operator()( int ix, int iy ) const {
            float radius = ceilf( 2.0f * sigma_ );

            float4 l = lfm_(ix, iy);
            float phi = atan2f(l.y, l.x);
            float a = radius * l.z;
            float b = radius * l.w;

            float cos_phi = cosf(phi);
            float sin_phi = sinf(phi);
            float3 uuT = make_float3(cos_phi*cos_phi, sin_phi*sin_phi, cos_phi*sin_phi);
            float3 vvT = make_float3(sin_phi*sin_phi, cos_phi*cos_phi, -cos_phi*sin_phi);
            float3 S = uuT / (a*a) + vvT / (b*b);

            int max_x = int(sqrtf(a*a * cos_phi*cos_phi +
                                  b*b * sin_phi*sin_phi));
            int max_y = int(sqrtf(a*a * sin_phi*sin_phi +
                                  b*b * cos_phi*cos_phi));

            T sum = make_zero<T>();
            float norm = 0;
            for (int j = -max_y; j <= max_y; ++j) {
                for (int i = -max_x; i <= max_x; ++i) {
                    float kernel = __expf(-(i*i*S.x + 2*i*j*S.z + j*j*S.y));
                    T c = src_(ix + i, iy + j);
                    sum += kernel * c;
                    norm += kernel;
                }
            }
            sum /=  norm;

            return sum;
        }
    };
}


oz::gpu_image oz::aniso_gauss( const gpu_image& src, const gpu_image& lfm,
                               float sigma, float angle, float precision  )
{
    if (sigma <= 0) return src;
    switch (src.format()) {
        case FMT_FLOAT:  return generate(src.size(), imp_aniso_gauss<float> (src, lfm, sigma, precision));
        case FMT_FLOAT3: return generate(src.size(), imp_aniso_gauss<float3>(src, lfm, sigma, precision));
        default:
            OZ_INVALID_FORMAT();
    }
}
