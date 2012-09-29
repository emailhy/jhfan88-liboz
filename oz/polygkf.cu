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
#include <oz/polygkf.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>

namespace oz {

    struct PolyGKF4 : public generator<float3> {
        gpu_sampler<float3,0> src_;
        int radius_;
        float q_, threshold_, zeta_, eta_;

        PolyGKF4( const gpu_image& src, int radius, float q, float threshold, float zeta, float eta )
            : src_(src), radius_(radius), q_(q), threshold_(threshold), zeta_(zeta), eta_(eta) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            const int N = 4;
            float3 m[N];
            float3 s[N];
            float w[N];
            {
                const float3 c = src_(ix, iy);
                const float wx = 1.0f / float(N);
                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    m[k] =  c * wx;
                    s[k] = c * c * wx;
                    w[k] = wx;
                }
            }

            for (int j = 0; j <= radius_; ++j) {
                for (int i = -radius_; i <= radius_; ++i) {
                    if ((j !=0) || (i > 0)) {
                        float2 v = make_float2(i , j) / radius_;

                        float dot_v = dot(v,v);
                        if (dot_v <= 1.0f) {
                            const float3 c0 = src_(ix + i, iy +j);
                            const float3 c1 = src_(ix - i, iy -j);

                            const float3 cc0 = c0 * c0;
                            const float3 cc1 = c1 * c1;

                            float n = 0;
                            float wx[N];
                            float z, xx, yy;

                            xx = zeta_ - eta_ * v.x * v.x;
                            yy = zeta_ - eta_ * v.y * v.y;
                            z = fmaxf(0,  v.y + xx);  n += wx[0] = z * z;
                            z = fmaxf(0, -v.x + yy);  n += wx[1] = z * z;
                            z = fmaxf(0, -v.y + xx);  n += wx[2] = z * z;
                            z = fmaxf(0,  v.x + yy);  n += wx[3] = z * z;

                            const float g = __expf(-3.125f * dot_v) / n;

                            #pragma unroll
                            for (int k0 = 0; k0 < N; ++k0) {
                                const float wk = wx[k0] * g;
                                const int k1 = (k0+(N/2))&(N-1);

                                m[k0] += c0 * wk;
                                s[k0] += cc0 * wk;
                                w[k0] += wk;

                                m[k1] += c1 * wk;
                                s[k1] += cc1 * wk;
                                w[k1] += wk;
                            }
                        }
                    }
                }
            }

            float3 o = make_float3(0);
            float ow = 0;
            #pragma unroll
            for (int k = 0; k < N; ++k ) {
                m[k] /= w[k];
                s[k] = fabs(s[k] / w[k] - m[k] * m[k]);
                float sigma2 = fmaxf(threshold_, sqrtf(sum(s[k])));
                float wk = __powf(sigma2, -q_);
                o += m[k] * wk;
                ow += wk;
            }
            return o /ow;
        }
    };


    struct PolyGKF8 : public generator<float3> {
        gpu_sampler<float3,0> src_;
        int radius_;
        float q_, threshold_, zeta_, eta_;

        PolyGKF8( const gpu_image& src, int radius, float q, float threshold, float zeta, float eta )
            : src_(src), radius_(radius), q_(q), threshold_(threshold), zeta_(zeta), eta_(eta) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            const int N = 8;

            float3 m[N];
            float3 s[N];
            float w[N];
            {
                const float3 c = src_(ix, iy);
                const float wx = 1.0f / float(N);
                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    m[k] =  c * wx;
                    s[k] = c * c * wx;
                    w[k] = wx;
                }
            }

            for (int j = 0; j <= radius_; ++j) {
                for (int i = -radius_; i <= radius_; ++i) {
                    if ((j !=0) || (i > 0)) {
                        float2 v = make_float2(i , j) / radius_;

                        float dot_v = dot(v,v);
                        if (dot_v <= 1.0f) {
                            const float3 c0 = src_(ix + i, iy +j);
                            const float3 c1 = src_(ix - i, iy -j);

                            const float3 cc0 = c0 * c0;
                            const float3 cc1 = c1 * c1;

                            float n = 0;
                            float wx[N];
                            float z, xx, yy;

                            xx = zeta_ - eta_ * v.x * v.x;
                            yy = zeta_ - eta_ * v.y * v.y;
                            z = fmaxf(0,  v.y + xx);  n += wx[0] = z * z;
                            z = fmaxf(0, -v.x + yy);  n += wx[2] = z * z;
                            z = fmaxf(0, -v.y + xx);  n += wx[4] = z * z;
                            z = fmaxf(0,  v.x + yy);  n += wx[6] = z * z;

                            v = CUDART_SQRT_HALF_F * make_float2( v.x - v.y, v.x + v.y );

                            xx = zeta_ - eta_ * v.x * v.x;
                            yy = zeta_ - eta_ * v.y * v.y;
                            z = fmaxf(0,  v.y + xx);  n += wx[1] = z * z;
                            z = fmaxf(0, -v.x + yy);  n += wx[3] = z * z;
                            z = fmaxf(0, -v.y + xx);  n += wx[5] = z * z;
                            z = fmaxf(0,  v.x + yy);  n += wx[7] = z * z;

                            const float g = __expf(-3.125f * dot_v) / n;

                            #pragma unroll
                            for (int k0 = 0; k0 < N; ++k0) {
                                const float wk = wx[k0] * g;
                                const int k1 = (k0+(N/2))&(N-1);

                                m[k0] += c0 * wk;
                                s[k0] += cc0 * wk;
                                w[k0] += wk;

                                m[k1] += c1 * wk;
                                s[k1] += cc1 * wk;
                                w[k1] += wk;
                            }
                        }
                    }
                }
            }

            float3 o = make_float3(0);
            float ow = 0;
            #pragma unroll
            for (int k = 0; k < N; ++k ) {
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


    gpu_image polygkf( const gpu_image& src, int N, int radius, float q,
                       float threshold, float zeta, float eta )
    {
        switch (N) {
            case 4: return generate(src.size(), PolyGKF4(src, radius, q, threshold, zeta, eta));
            case 8: return generate(src.size(), PolyGKF8(src, radius, q, threshold, zeta, eta));
            default:
                OZ_X() << "Unsupported parameter value (N=" << N << ")";
        }
    }
}
