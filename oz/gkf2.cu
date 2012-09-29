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
#include <oz/gkf2.h>
#include <oz/gpu_plm2.h>
#include <oz/gpu_binder.h>
#include <oz/launch_config.h>

namespace oz {

    static texture<float4, 2> texSRC;
    static texture<float, 2> texT0; // filterMode=Linear, addressMode=Wrap, normalized=true

    template <int N>
    __global__ void gkf_filter( gpu_plm2<float3> dst, float radius, float q, float threshold ) {
        int ix = blockDim.x * blockIdx.x + threadIdx.x;
        int iy = blockDim.y * blockIdx.y + threadIdx.y;
        if (ix >= dst.w || iy >= dst.h) return;

        float3 m[N];
        float3 s[N];
        float w[N];
        #pragma unroll
        for (int k = 0; k < N; ++k) {
            m[k] = s[k] = make_float3(0);
            w[k] = 0;
        }

        float piN = -2 * CUDART_PI_F / N;
        float4 RpiN = make_float4( cosf(piN), sinf(piN),
                                  -sinf(piN), cosf(piN) );

        int r = (int)ceilf(radius);
        for (int j = -r; j <= r; ++j) {
            for (int i = -r; i <= r; ++i) {
                float2 v = make_float2( 0.5f * i / radius,
                                        0.5f * j / radius);

                if (dot(v,v) <= 0.25f) {
                    float3 c = make_float3(tex2D(texSRC, ix + i, iy + j));
                    float3 cc = c * c;

                    #pragma unroll
                    for (int k = 0; k < N; ++k) {
                        float wx = tex2D(texT0, v.x, v.y);

                        m[k] += c * wx;
                        s[k] += cc * wx;
                        w[k] += wx;

                        v = make_float2( RpiN.x * v.x + RpiN.z * v.y,
                                         RpiN.y * v.x + RpiN.w * v.y );
                    }
                }
            }
        }

        float3 o = make_float3(0);
        float ow = 0;
        #pragma unroll
        for (int k = 0; k < N; ++k) {
            m[k] /= w[k];
            s[k] = fabs(s[k] / w[k] - m[k] * m[k]);
            float sigma2 = fmaxf(threshold, sqrtf(s[k].x + s[k].y + s[k].z));
            float wk = powf(sigma2, -q);
            o += m[k] * wk;
            ow += wk;
        }
        dst.write(ix, iy, o / ow);
    }


    gpu_image gkf_filter2( const gpu_image& src, const gpu_image& krnl,
                           float radius, float q, float threshold )
    {
        gpu_binder<float3> src_(texSRC, src);
        gpu_binder<float> krnl_(texT0, krnl, cudaFilterModeLinear, cudaAddressModeWrap, true);
        gpu_image dst(src.size(), FMT_FLOAT3);
        launch_config cfg(dst);
        gkf_filter<8><<<cfg.blocks(), cfg.threads()>>>(dst, radius, q, threshold);
        return dst;
    }

}
