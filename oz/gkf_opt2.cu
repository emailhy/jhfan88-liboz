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
#include <oz/gkf_opt2.h>
#include <oz/generate.h>
#include <oz/gpu_sampler2.h>

namespace oz {

    template<typename T> struct imp_gkf_opt82 : public generator<T> {
        gpu_sampler<T,0> src_;
        gpu_sampler<float4,1> krnl_;
        float radius_;
        float q_;
        float threshold_;

        imp_gkf_opt82( const gpu_image& src, const gpu_image& krnl, float radius, float q, float threshold )
            : src_(src), krnl_(krnl,cudaFilterModeLinear, cudaAddressModeWrap, true),
              radius_(radius), q_(q), threshold_(threshold) {}

        inline __device__ T operator()( int ix, int iy) const {
            T o = make_zero<T>();
            float ow = 0;
            for (int rr = 0; rr < 2; ++rr ) {
                T m[4];
                T s[4];
                float w[4];
                {
                    T c = src_(ix, iy);
                    float wx = krnl_(0,0).x;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        m[k] =  c * wx;
                        s[k] = c * c * wx;
                        w[k] = wx;
                    }
                }

                int r = (int)ceilf(radius_);
                for (int j = 0; j <= r; ++j) {
                    for (int i = -r; i <= r; ++i) {
                        if ((j !=0) || (i > 0)) {
                        float2 v = make_float2( 0.5f * i / radius_, 0.5f * j / radius_);
                            if (rr > 0)
                                v = CUDART_SQRT_HALF_F * make_float2( v.x - v.y, v.x + v.y );

                            const float dot_v = dot(v,v);
                            if (dot_v <= 0.25f) {
                                const float4 tmp = krnl_(v);
                                float const wx[4] = { tmp.x, tmp.y, tmp.z, tmp.w };

                                T c = src_(ix + i, iy + j);
                                T cc = c * c;
                                #pragma unroll
                                for (int k = 0; k < 4; ++k) {
                                    const float wk = wx[k];
                                    m[k] += wk * c;
                                    s[k] += wk * cc;
                                    w[k] += wk;
                                }

                                c = src_(ix - i, iy - j);
                                cc = c * c;
                                #pragma unroll
                                for (int k = 0; k < 4; ++k) {
                                    const float wk = wx[(k + 2) & 3];
                                    m[k] += wk * c;
                                    s[k] += wk * cc;
                                    w[k] += wk;
                                }
                            }
                        }
                    }
                }

                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    m[k] /= w[k];
                    s[k] = fabs(s[k] / w[k] - m[k] * m[k]);
                    float sigma2 = fmaxf(threshold_, sqrtf(sum(s[k])));
                    float wk = __powf(sigma2, -q_);
                    o += m[k] * wk;
                    ow += wk;
                }
            }
            return o / ow;
        }
    };


    gpu_image gkf_opt8_filter2( const gpu_image& src, const gpu_image& krnl,
                                float radius, float q, float threshold )
    {
        switch (src.format()) {
            case FMT_FLOAT:  return generate(src.size(), imp_gkf_opt82<float >(src, krnl, radius, q, threshold));
            case FMT_FLOAT3: return generate(src.size(), imp_gkf_opt82<float3>(src, krnl, radius, q, threshold));
            default:
                OZ_INVALID_FORMAT();
        }
    }

}
