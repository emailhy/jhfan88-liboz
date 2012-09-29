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
#include <oz/akf_opt3.h>
#include <oz/gpu_plm2.h>
#include <oz/gpu_binder.h>
#include <oz/launch_config.h>
#include <oz/st_util.h>

namespace oz {

    static texture<float4, 2> texSRC;   // filterMode=Point, addressMode=Clamp, normalized=false
    static texture<float4, 2> texH0246; // filterMode=Linear, addressMode=Wrap, normalized=true

    __global__ void akf_filter3( gpu_plm2<float3> dst, const gpu_plm2<float3> st, float radius,
                                 float q, float alpha, float threshold )
    {
        const int ix = blockDim.x * blockIdx.x + threadIdx.x;
        const int iy = blockDim.y * blockIdx.y + threadIdx.y;
        if (ix >= dst.w || iy >= dst.h) return;

        const float3 t = st2tA(st(ix, iy));
        const float a = radius * clamp((alpha + t.z) / alpha, 0.1f, 2.0f);
        const float b = radius * clamp(alpha / (alpha + t.z), 0.1f, 2.0f);
        const int max_x = int(sqrtf(a*a * t.x*t.x + b*b * t.y*t.y));
        const int max_y = int(sqrtf(a*a * t.y*t.y + b*b * t.x*t.x));
        float4 SR = 0.5f * make_float4(t.x/a, t.y/a, -t.y/b, t.x/b);

        float3 o = make_float3(0);
        float ow = 0;
        for (int rr = 0; rr < 2; ++rr) {
            float3 m[4];
            float3 s[4];
            float w[4];
            {
                float3 c = make_float3(tex2D(texSRC, ix, iy));
                float wx = tex2D(texH0246, 0, 0).x;
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
                        float2 v = make_float2( SR.x * i + SR.y * j, SR.z * i + SR.w * j );

                        const float dot_v = dot(v,v);
                        if (dot_v <= 0.25f) {
                            const float4 tmp = tex2D(texH0246, v.x, v.y);
                            float const wx[4] = { tmp.x, tmp.y, tmp.z, tmp.w };

                            float3 c = make_float3(tex2D(texSRC, ix + i, iy + j));
                            float3 cc = c * c;
                            #pragma unroll
                            for (int k = 0; k < 4; ++k) {
                                const float wk = wx[k];
                                m[k] += wk * c;
                                s[k] += wk * cc;
                                w[k] += wk;
                            }

                            c = make_float3(tex2D(texSRC, ix - i, iy - j));
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
                float sigma2 = fmaxf(threshold, sqrtf(sum(s[k])));
                float alpha_k = __powf(sigma2, -q);
                o += m[k] * alpha_k;
                ow += alpha_k;
            }

            SR = CUDART_SQRT_HALF_F * make_float4( SR.x - SR.z, SR.y - SR.w,
                                                   SR.x + SR.z, SR.y + SR.w );
        }
        dst.write(ix, iy, o / ow);
    }


    gpu_image akf_opt_filter3( const gpu_image& src, const gpu_image& st, const gpu_image& krnl,
                               float radius, float q, float alpha, float threshold )
    {
        gpu_binder<float3> src_(texSRC, src);
        gpu_binder<float4> krnl_(texH0246, krnl, cudaFilterModeLinear, cudaAddressModeWrap, true);
        gpu_image dst(src.size(), FMT_FLOAT3);
        launch_config cfg(dst);
        akf_filter3<<<cfg.blocks(), cfg.threads()>>>(dst, st, radius, q, alpha, threshold);
        return dst;
    }
}

