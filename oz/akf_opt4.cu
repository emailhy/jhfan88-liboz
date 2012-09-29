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
#include <oz/akf_opt4.h>
#include <oz/gpu_plm2.h>
#include <oz/gpu_binder.h>
#include <oz/launch_config.h>
#include <oz/st_util.h>

namespace oz {

    static texture<float4, 2> texSRC;   // filterMode=Point, addressMode=Clamp, normalized=false
    static texture<float4, 2> texH0123; // filterMode=Linear, addressMode=Wrap, normalized=true

    __global__ void akf_filter4( gpu_plm2<float3> dst, const gpu_plm2<float3> st, float radius,
                                 float q, float alpha, float threshold, float a_star )
    {
        const int ix = blockDim.x * blockIdx.x + threadIdx.x;
        const int iy = blockDim.y * blockIdx.y + threadIdx.y;
        if (ix >= dst.w || iy >= dst.h) return;

        float3 t = st2tA(st(ix, iy), a_star);
        float a = radius * clamp((alpha + t.z) / alpha, 0.1f, 2.0f);
        float b = radius * clamp(alpha / (alpha + t.z), 0.1f, 2.0f);
        int max_x = int(sqrtf(a*a * t.x*t.x + b*b * t.y*t.y));
        int max_y = int(sqrtf(a*a * t.y*t.y + b*b * t.x*t.x));
        float4 SR = 0.5f * make_float4(t.x/a, t.y/a, -t.y/b, t.x/b);

        float3 m[8];
        float3 s[8];
        float w[8];
        {
            const float3 c = make_float3(tex2D(texSRC, ix, iy));
            const float wx = tex2D(texH0123, 0, 0).x;
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
                    float2 v = make_float2( SR.x * i + SR.y * j, SR.z * i + SR.w * j );

                    float dot_v = dot(v,v);
                    if (dot_v <= 0.25f) {
                        float3 c0 = make_float3(tex2D(texSRC, ix + i, iy + j));
                        float3 c1 = make_float3(tex2D(texSRC, ix - i, iy - j));
                        float3 cc0 = c0 * c0;
                        float3 cc1 = c1 * c1;

                        float4 tmp0 = tex2D(texH0123, v.x, v.y);
                        float4 tmp1 = tex2D(texH0123, -v.x, -v.y);
                        float const wx[8] = { tmp0.x, tmp0.y, tmp0.z, tmp0.w,
                                              tmp1.x, tmp1.y, tmp1.z, tmp1.w };

                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            m[k] += wx[k] * c0  + wx[(k+4)&7] * c1;
                            s[k] += wx[k] * cc0 + wx[(k+4)&7] * cc1;
                            w[k] += wx[k]       + wx[(k+4)&7];
                        }
                    }
                }
            }
        }

        float3 o = make_float3(0);
        float ow = 0;
        #pragma unroll
        for (int k = 0; k < 8; ++k ) {
            m[k] /= w[k];
            s[k] = fabs(s[k] / w[k] - m[k] * m[k]);
            float sigma2 = fmaxf(threshold, sqrtf(sum(s[k])));
            float alpha_k = __powf(sigma2, -q);
            o += m[k] * alpha_k;
            ow += alpha_k;
        }
        dst.write(ix, iy, o / ow);
    }


    gpu_image akf_opt_filter4( const gpu_image& src, const gpu_image& st, const gpu_image& krnl,
                               float radius, float q, float alpha, float threshold, float a_star )
    {
        gpu_binder<float3> src_(texSRC, src);
        gpu_binder<float4> krnl_(texH0123, krnl, cudaFilterModeLinear, cudaAddressModeWrap, true);
        gpu_image dst(src.size(), FMT_FLOAT3);
        launch_config cfg(dst);
        akf_filter4<<<cfg.blocks(), cfg.threads()>>>(dst, st, radius, q, alpha, threshold, a_star);
        return dst;
    }
}

