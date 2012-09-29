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
#include <oz/gkf_opt.h>
#include <oz/generate.h>
#include <oz/gpu_sampler2.h>

namespace oz {

    template<typename T> struct imp_gkf_opt8 : public generator<T> {
        gpu_sampler<T,0> src_;
        gpu_sampler<float4,1> krnl_;
        float radius_;
        float q_;
        float threshold_;

        imp_gkf_opt8( const gpu_image& src, const gpu_image& krnl, float radius, float q, float threshold )
            : src_(src), krnl_(krnl,cudaFilterModeLinear, cudaAddressModeWrap, true),
              radius_(radius), q_(q), threshold_(threshold) {}

        inline __device__ T operator()( int ix, int iy) const {
            T m[8];
            T s[8];
            float w[8];
            {
                T c = src_(ix, iy);
                float wx = krnl_(0,0).x;
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    m[k] =  c * wx;
                    s[k] = c * c * wx;
                    w[k] = wx;
                }
            }

            float piN = 2 * CUDART_PI_F / 8;
            float4 RpiN = make_float4(cosf(piN), sinf(piN), -sinf(piN), cosf(piN));

            int r = (int)ceilf(radius_);
            for (int j = 0; j <= r; ++j) {
                for (int i = -r; i <= r; ++i) {
                    if ((j !=0) || (i > 0)) {
                    float2 v = make_float2( 0.5f * i / radius_,
                                            0.5f * j / radius_);

                        float dot_v = dot(v,v);
                        if (dot_v <= 0.25f) {
                            T c0 = src_(ix + i, iy +j);
                            T c1 = src_(ix - i, iy -j);

                            T cc0 = c0 * c0;
                            T cc1 = c1 * c1;

                            float4 tmp0 = krnl_(v);
                            float4 tmp1 = krnl_(-v);
                            float wx[8] = { tmp0.x, tmp0.y, tmp0.z, tmp0.w,
                                            tmp1.x, tmp1.y, tmp1.z, tmp1.w };

                            #pragma unroll
                            for (int k = 0; k < 4; ++k) {
                                m[k] += c0 * wx[k] + c1 * wx[k+4];
                                s[k] += cc0 * wx[k] + cc1 * wx[k+4];
                                w[k] += wx[k] + wx[k+4];

                                m[k+4] += c1 * wx[k] + c0 * wx[k+4];
                                s[k+4] += cc1 * wx[k] + cc0 * wx[k+4];
                                w[k+4] += wx[k] + wx[k+4];
                            }
                        }
                    }
                }
            }

            T o = make_zero<T>();
            float ow = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                m[k] /= w[k];
                s[k] = fabs(s[k] / w[k] - m[k] * m[k]);
                float sigma2 = fmaxf(threshold_, sqrtf(sum(s[k])));
                float wk = __powf(sigma2, -q_);
                o += m[k] * wk;
                ow += wk;
            }
            return o / ow;
        }
    };


    gpu_image gkf_opt8_filter( const gpu_image& src, const gpu_image& krnl,
                               float radius, float q, float threshold )
    {
        switch (src.format()) {
            case FMT_FLOAT:  return generate(src.size(), imp_gkf_opt8<float >(src, krnl, radius, q, threshold));
            case FMT_FLOAT3: return generate(src.size(), imp_gkf_opt8<float3>(src, krnl, radius, q, threshold));
            default:
                OZ_INVALID_FORMAT();
        }
    }

}
