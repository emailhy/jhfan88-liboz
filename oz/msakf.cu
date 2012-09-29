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
#include <oz/msakf.h>
#include <oz/transform.h>
#include <oz/generate.h>
#include <oz/st_util.h>
#include <oz/gpu_sampler3.h>

namespace oz {

    struct MsAkfPropagate : public ternary_function<float4,float4,float,float4> {
        float scale_;

        MsAkfPropagate( float scale ) : scale_(scale) {}

        inline __device__ float4 operator()(float4 a, float4 b, float w) const {
            float u = __saturatef( scale_ * w );
            return a * u + (1 - u) * b;
        }
    };

    gpu_image msakf_propagate( const gpu_image& A, const gpu_image& B,
                               const gpu_image& w, float scale )
    {
        return transform(A, B, w, MsAkfPropagate(scale));
    }



    struct imp_msakf_single : public generator<float4> {
        gpu_sampler<float4,0> src_;
        gpu_sampler<float3,1> st_;
        gpu_sampler<float4,2> krnl_;
        float radius_;
        float q_;
        float alpha_;
        float threshold_;

        imp_msakf_single( const gpu_image& src, const gpu_image& st,
                          const gpu_image& krnl, float radius, float q, float alpha, float threshold )
            : src_(src), st_(st), krnl_(krnl, cudaFilterModeLinear, cudaAddressModeWrap, true),
              radius_(radius), q_(q), alpha_(alpha), threshold_(threshold) {}

        inline __device__ float4 operator()( int ix, int iy ) const {
            const int N = 8;
            float3 m[8];
            float3 s[8];
            float w[8];
            {
                float3 c = make_float3(src_(ix, iy));
                float wx = krnl_(0,0).x;
                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    m[k] =  c * wx;
                    s[k] = c * c * wx;
                    w[k] = wx;
                }
            }

            float3 t = st2tA(st_(ix, iy));
            float a = radius_ * clamp((alpha_ + t.z) / alpha_, 0.1f, 2.0f);
            float b = radius_ * clamp(alpha_ / (alpha_ + t.z), 0.1f, 2.0f);
            int max_x = int(sqrtf(a*a * t.x*t.x + b*b * t.y*t.y));
            int max_y = int(sqrtf(a*a * t.y*t.y + b*b * t.x*t.x));
            float4 SR = 0.5f * make_float4(t.x/a, -t.y/b, t.y/a, t.x/b);

            for (int j = 0; j <= max_y; ++j) {
                for (int i = -max_x; i <= max_x; ++i) {
                    if ((j !=0) || (i > 0)) {
                        float2 v = make_float2(SR.x * i + SR.z * j,
                                               SR.y * i + SR.w * j);

                        float dot_v = dot(v,v);
                        if (dot_v <= 0.25f) {
                            float3 c0 = make_float3(src_(ix + i, iy +j));
                            float3 c1 = make_float3(src_(ix - i, iy -j));

                            float3 cc0 = c0 * c0;
                            float3 cc1 = c1 * c1;

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

            float3 o = make_float3(0);
            float ow = 0;
            float stotal2 = 0;
            #pragma unroll
            for (int k = 0; k < N; ++k ) {
                m[k] /= w[k];
                s[k] = fabs(s[k] / w[k] - m[k] * m[k]);
                float sigma2 = fmaxf(threshold_, sqrtf(sum(s[k])));
                float wk = __powf(sigma2, -q_);
                o += m[k] * wk;
                ow += wk;
                stotal2 += sigma2;
            }
            return make_float4(o / ow, stotal2);
        }
    };


    gpu_image msakf_single( const gpu_image& src, const gpu_image& st, const gpu_image& krnl,
                            float radius, float q, float alpha, float threshold )
    {
        return generate(src.size(), imp_msakf_single(src, st, krnl, radius, q, alpha, threshold));
    }
}