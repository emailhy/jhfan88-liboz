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
#include <oz/akf_opt.h>
#include <oz/generate.h>
#include <oz/gpu_plm2.h>
#include <oz/gpu_sampler3.h>
#include <oz/st_util.h>

namespace oz {

    template <typename T> struct imp_akf_opt4 : public generator<T> {
        gpu_sampler<T,0> src_;
        gpu_sampler<float3,1> st_;
        gpu_sampler<float4,2> krnl_;
        float radius_;
        float q_;
        float alpha_;
        float threshold_;

        imp_akf_opt4( const gpu_image& src, const gpu_image& st,
                      const gpu_image& krnl, float radius, float q, float alpha, float threshold )
            : src_(src), st_(st), krnl_(krnl, cudaFilterModeLinear, cudaAddressModeWrap, true),
              radius_(radius), q_(q), alpha_(alpha), threshold_(threshold) {}

        inline __device__ T operator()( int ix, int iy ) const {
            const float3 t = st2tA(st_(ix, iy));
            const float a = radius_ * clamp((alpha_ + t.z) / alpha_, 0.1f, 2.0f);
            const float b = radius_ * clamp(alpha_ / (alpha_ + t.z), 0.1f, 2.0f);
            const int max_x = int(sqrtf(a*a * t.x*t.x + b*b * t.y*t.y));
            const int max_y = int(sqrtf(a*a * t.y*t.y + b*b * t.x*t.x));
            const float4 SR = 0.5f * make_float4(t.x/a, -t.y/b, t.y/a, t.x/b);

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
                        const float2 v = make_float2(SR.x * i + SR.z * j, SR.y * i + SR.w * j);
                        if (dot(v,v) <= 0.25f) {
                            const T c0 = src_(ix + i, iy + j);
                            const T c1 = src_(ix - i, iy - j);
                            const T cc0 = c0 * c0;
                            const T cc1 = c1 * c1;

                            const float4 tmp = krnl_(v);
                            float const wx[4] = { tmp.x, tmp.y, tmp.z, tmp.w };

                            #pragma unroll
                            for (int k = 0; k < 2; ++k) {
                                m[k] += wx[k] * c0  + wx[k+2] * c1;
                                s[k] += wx[k] * cc0 + wx[k+2] * cc1;
                                w[k] += wx[k]       + wx[k+2];

                                m[k+2] += wx[k] * c1  + wx[k+2] * c0;
                                s[k+2] += wx[k] * cc1 + wx[k+2] * cc0;
                                w[k+2] += wx[k]       + wx[k+2];
                            }
                        }
                    }

                }
            }

            T o = make_zero<T>();
            float ow = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k ) {
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


    template <typename T> struct imp_akf_opt8 : public generator<T> {
        gpu_sampler<T,0> src_;
        gpu_sampler<float3,1> st_;
        gpu_sampler<float4,2> krnl_;
        float radius_;
        float q_;
        float alpha_;
        float threshold_;

        imp_akf_opt8( const gpu_image& src, const gpu_image& st,
                      const gpu_image& krnl, float radius, float q, float alpha, float threshold )
            : src_(src), st_(st), krnl_(krnl, cudaFilterModeLinear, cudaAddressModeWrap, true),
              radius_(radius), q_(q), alpha_(alpha), threshold_(threshold) {}

        inline __device__ T operator()( int ix, int iy ) const {
            const float3 t = st2tA(st_(ix, iy));
            const float a = radius_ * clamp((alpha_ + t.z) / alpha_, 0.1f, 2.0f);
            const float b = radius_ * clamp(alpha_ / (alpha_ + t.z), 0.1f, 2.0f);
            const int max_x = int(sqrtf(a*a * t.x*t.x + b*b * t.y*t.y));
            const int max_y = int(sqrtf(a*a * t.y*t.y + b*b * t.x*t.x));
            const float4 SR = 0.5f * make_float4(t.x/a, -t.y/b, t.y/a, t.x/b);

            T m[8];
            T s[8];
            float w[8];
            {
                const T c = src_(ix, iy);
                const float wx = krnl_(0,0).x;
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    m[k] =  c * wx;
                    s[k] = c * c * wx;
                    w[k] = wx;
                }
            }

            for (int j = 0; j <= max_y; ++j) {
                for (int i = -max_x; i <= max_x; ++i) {
                    if ((j !=0) || (i > 0)) {
                        float2 v = make_float2(SR.x * i + SR.z * j, SR.y * i + SR.w * j);

                        float dot_v = dot(v,v);
                        if (dot_v <= 0.25f) {
                            const T c0 = src_(ix + i, iy +j);
                            const T c1 = src_(ix - i, iy -j);
                            const T cc0 = c0 * c0;
                            const T cc1 = c1 * c1;

                            const float4 tmp0 = krnl_(v);
                            const float4 tmp1 = krnl_(-v);
                            float const wx[8] = { tmp0.x, tmp0.y, tmp0.z, tmp0.w,
                                                  tmp1.x, tmp1.y, tmp1.z, tmp1.w };

                            #pragma unroll
                            for (int k = 0; k < 4; ++k) {
                                m[k] += wx[k] * c0  + wx[k+4] * c1;
                                s[k] += wx[k] * cc0 + wx[k+4] * cc1;
                                w[k] += wx[k]       + wx[k+4];

                                m[k+4] += wx[k] * c1  + wx[k+4] * c0;
                                s[k+4] += wx[k] * cc1 + wx[k+4] * cc0;
                                w[k+4] += wx[k]       + wx[k+4];
                            }
                        }
                    }
                }
            }

            T o = make_zero<T>();
            float ow = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k ) {
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


    gpu_image akf_opt_filter( const gpu_image& src, const gpu_image& st, const gpu_image& krnl,
                              float radius, float q, float alpha, float threshold, int N )
    {
        if ((N != 4) && (N != 8)) OZ_X() << "Invalid N!";
        switch (src.format()) {
            case FMT_FLOAT:  if (N == 4) return generate(src.size(), imp_akf_opt4<float >(src, st, krnl, radius, q, alpha, threshold));
                                    else return generate(src.size(), imp_akf_opt8<float >(src, st, krnl, radius, q, alpha, threshold));
            case FMT_FLOAT3: if (N == 4) return generate(src.size(), imp_akf_opt4<float3>(src, st, krnl, radius, q, alpha, threshold));
                                    else return generate(src.size(), imp_akf_opt8<float3>(src, st, krnl, radius, q, alpha, threshold));
            default:
                OZ_INVALID_FORMAT();
        }
    }
}

