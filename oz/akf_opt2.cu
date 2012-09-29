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
#include <oz/akf_opt2.h>
#include <oz/generate.h>
#include <oz/gpu_plm2.h>
#include <oz/gpu_sampler3.h>
#include <oz/st_util.h>

namespace oz {

    template <typename T> struct imp_akf_opt82 : public generator<T> {
        gpu_sampler<T,0> src_;
        gpu_sampler<float3,1> st_;
        gpu_sampler<float4,2> krnl_;
        float radius_;
        float q_;
        float alpha_;
        float threshold_;

        imp_akf_opt82( const gpu_image& src, const gpu_image& st,
                       const gpu_image& krnl, float radius, float q, float alpha, float threshold )
            : src_(src), st_(st), krnl_(krnl, cudaFilterModeLinear, cudaAddressModeWrap, true),
              radius_(radius), q_(q), alpha_(alpha), threshold_(threshold) {}

        inline __device__ T operator()( int ix, int iy ) const {
            const float3 t = st2tA(st_(ix, iy));
            const float a = radius_ * clamp((alpha_ + t.z) / alpha_, 0.1f, 2.0f);
            const float b = radius_ * clamp(alpha_ / (alpha_ + t.z), 0.1f, 2.0f);
            const int max_x = int(sqrtf(a*a * t.x*t.x + b*b * t.y*t.y));
            const int max_y = int(sqrtf(a*a * t.y*t.y + b*b * t.x*t.x));
            float4 SR = 0.5f * make_float4(t.x/a, -t.y/b, t.y/a, t.x/b);

            T o = make_zero<T>();
            float ow = 0;
            for (int rr = 0; rr < 2; ++rr) {
                T m[4];
                T s[4];
                float w[4];
                {
                    const T c = src_(ix, iy);
                    const float wx = krnl_(0,0).x;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        m[k] =  c * wx;
                        s[k] = c * c * wx;
                        w[k] = wx;
                    }
                }

                for (int j = 0; j <= max_y; ++j) {
                    for (int i = -max_x; i <= max_x; ++i) {
                        if ((j !=0) || (i > 0)) {
                            const float2 v = make_float2(SR.x * i + SR.z * j,
                                                         SR.y * i + SR.w * j);

                            const float dot_v = dot(v,v);
                            if (dot_v <= 0.25f) {
                                const float4 tmp = krnl_(v);
                                float const wx[4] = { tmp.x, tmp.y, tmp.z, tmp.w };

                                {
                                    const T c = src_(ix + i, iy + j);
                                    const T cc = c * c;
                                    #pragma unroll
                                    for (int k = 0; k < 4; ++k) {
                                        const float wk = wx[k];
                                        m[k] += wk * c;
                                        s[k] += wk * cc;
                                        w[k] += wk;
                                    }
                                }
                                {
                                    const T c = src_(ix - i, iy - j);
                                    const T cc = c * c;
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
                }

                #pragma unroll
                for (int k = 0; k < 4; ++k ) {
                    m[k] /= w[k];
                    s[k] = fabs(s[k] / w[k] - m[k] * m[k]);
                    float sigma2 = fmaxf(threshold_, sqrtf(sum(s[k])));
                    float wk = __powf(sigma2, -q_);
                    o += m[k] * wk;
                    ow += wk;
                }

                SR = CUDART_SQRT_HALF_F * make_float4( SR.x - SR.y, SR.x + SR.y,
                                                       SR.z - SR.w, SR.z + SR.w );
            }
            return o / ow;
        }
    };


    gpu_image akf_opt_filter2( const gpu_image& src, const gpu_image& st, const gpu_image& krnl,
                               float radius, float q, float alpha, float threshold )
    {
        switch (src.format()) {
            case FMT_FLOAT:  return generate(src.size(), imp_akf_opt82<float >(src, st, krnl, radius, q, alpha, threshold));
            case FMT_FLOAT3: return generate(src.size(), imp_akf_opt82<float3>(src, st, krnl, radius, q, alpha, threshold));
            default:
                OZ_INVALID_FORMAT();
        }
    }
}

