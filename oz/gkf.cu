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
#include <oz/gkf.h>
#include <oz/generate.h>
#include <oz/gpu_sampler2.h>

namespace oz {

    template<typename T, int N> struct imp_gkf : public generator<T> {
        gpu_sampler<T,0> src_;
        gpu_sampler<float,1> krnl_;
        float radius_;
        float q_;
        float threshold_;

        imp_gkf( const gpu_image& src, const gpu_image& krnl, float radius, float q, float threshold )
            : src_(src), krnl_(krnl,cudaFilterModeLinear, cudaAddressModeWrap, true),
              radius_(radius), q_(q), threshold_(threshold) {}

        inline __device__ T operator()( int ix, int iy) const {
            T m[N];
            T s[N];
            float w[N];
            #pragma unroll
            for (int k = 0; k < N; ++k) {
                m[k] = make_zero<T>();
                s[k] = make_zero<T>();
                w[k] = 0;
            }

            float piN = 2 * CUDART_PI_F / float(N);
            float4 RpiN = make_float4(cosf(piN), sinf(piN), -sinf(piN), cosf(piN));

            int r = (int)ceilf(radius_);
            for (int j = -r; j <= r; ++j) {
                for (int i = -r; i <= r; ++i) {
                    float2 v = make_float2( 0.5f * i / radius_,
                                            0.5f * j / radius_);

                    if (dot(v,v) <= 0.25f) {
                        T c = src_(ix + i, iy + j);
                        T cc = c * c;

                        #pragma unroll
                        for (int k = 0; k < N; ++k) {
                            float wx = krnl_(v);

                            m[k] += c * wx;
                            s[k] += cc * wx;
                            w[k] += wx;

                            v = make_float2( RpiN.x * v.x + RpiN.z * v.y,
                                             RpiN.y * v.x + RpiN.w * v.y );
                        }
                    }
                }
            }

            T o = make_zero<T>();
            float ow = 0;
            #pragma unroll
            for (int k = 0; k < N; ++k) {
                m[k] /= w[k];
                s[k] = fabs(s[k] / w[k] - m[k] * m[k]);
                float sigma2 = fmaxf(threshold_, sqrtf(sum(s[k])));
                float alpha_k = __powf(sigma2, -q_);
                o += m[k] * alpha_k;
                ow += alpha_k;
            }
            return o / ow;
        }
    };


    gpu_image gkf_filter( const gpu_image& src, const gpu_image& krnl,
                          float radius, float q, float threshold, int N )
    {
        if ((N != 4) && (N != 8)) OZ_X() << "Invalid N!";
        switch (src.format()) {
            case FMT_FLOAT:
                return (N == 4)? generate(src.size(), imp_gkf<float ,4>(src, krnl, radius, q, threshold))
                               : generate(src.size(), imp_gkf<float ,8>(src, krnl, radius, q, threshold));
            case FMT_FLOAT3:
                return (N == 4)? generate(src.size(), imp_gkf<float3,4>(src, krnl, radius, q, threshold))
                               : generate(src.size(), imp_gkf<float3,8>(src, krnl, radius, q, threshold));
            default:
                OZ_INVALID_FORMAT();
        }
    }

}
