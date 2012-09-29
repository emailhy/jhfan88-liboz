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
#include <oz/polyakf_opt2.h>
#include <oz/generate.h>
#include <oz/gpu_plm2.h>
#include <oz/gpu_sampler2.h>
#include <oz/st_util.h>

namespace oz {

    struct imp_polyakf_opt82 : public generator<float3> {
        gpu_sampler<float3,0> src_;
        gpu_sampler<float3,1> st_;
        float radius_;
        float q_;
        float alpha_;
        float threshold_;
        float zeta_;
        /*float eta_;*/

        imp_polyakf_opt82( const gpu_image& src, const gpu_image& st, float radius, float q,
                           float alpha, float threshold, float zeta/*, float eta*/ )
            : src_(src), st_(st), radius_(radius), q_(q), alpha_(alpha),
              threshold_(threshold), zeta_(zeta)/*, eta_(eta)*/ {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            const float eta_ = 3.77f;

            const float3 t = st2tA(st_(ix, iy));
            const float a = radius_ * clamp((alpha_ + t.z) / alpha_, 0.1f, 2.0f);
            const float b = radius_ * clamp(alpha_ / (alpha_ + t.z), 0.1f, 2.0f);
            const int max_x = int(sqrtf(a*a * t.x*t.x + b*b * t.y*t.y));
            const int max_y = int(sqrtf(a*a * t.y*t.y + b*b * t.x*t.x));
            float4 SR = make_float4(t.x/a, -t.y/b, t.y/a, t.x/b);

            float3 o = make_zero<float3>();
            float ow = 0;
            for (int rr = 0; rr < 2; ++rr) {
                float3 m[4], s[4];
                float w[4];
                const float3 c = src_(ix, iy);
                const float wx = 1.0f / 8;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    m[k] =  c * wx;
                    s[k] = c * c * wx;
                    w[k] = wx;
                }

                for (int j = 0; j <= max_y; ++j) {
                    for (int i = -max_x; i <= max_x; ++i) {
                        if ((j !=0) || (i > 0)) {
                            float2 v = make_float2(SR.x * i + SR.z * j,
                                                   SR.y * i + SR.w * j);

                            float dot_v = dot(v,v);
                            if (dot_v <= 1.0f) {
                                float sum = 0;
                                float wx[4];
                                float z, xx, yy;

                                xx = zeta_ - eta_ * v.x * v.x;
                                yy = zeta_ - eta_ * v.y * v.y;
                                z = fmaxf(0,  v.y + xx);  sum += wx[0] = z * z;
                                z = fmaxf(0, -v.x + yy);  sum += wx[1] = z * z;
                                z = fmaxf(0, -v.y + xx);  sum += wx[2] = z * z;
                                z = fmaxf(0,  v.x + yy);  sum += wx[3] = z * z;

                                v = CUDART_SQRT_HALF_F * make_float2( v.x - v.y, v.x + v.y );

                                xx = zeta_ - eta_ * v.x * v.x;
                                yy = zeta_ - eta_ * v.y * v.y;
                                z = fmaxf(0,  v.y + xx);  sum += z * z;
                                z = fmaxf(0, -v.x + yy);  sum += z * z;
                                z = fmaxf(0, -v.y + xx);  sum += z * z;
                                z = fmaxf(0,  v.x + yy);  sum += z * z;

                                const float g = __expf(-3.125f * dot_v) / sum;
                                #pragma unroll
                                for (int k = 0; k < 4; ++k) wx[k] *= g;

                                float3 c = src_(ix + i, iy + j);
                                float3 cc = c * c;
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


    gpu_image polyakf_opt_filter2( const gpu_image& src, const gpu_image& st, float radius,
                                   float q, float alpha, float threshold, float zeta, float eta )
    {
        return generate(src.size(), imp_polyakf_opt82(src, st, radius, q, alpha, threshold, zeta/*, eta*/));
    }
}

