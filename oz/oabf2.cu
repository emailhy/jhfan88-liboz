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
#include <oz/oabf2.h>
#include <oz/gpu_plm2.h>
#include <oz/gpu_binder.h>
#include <oz/launch_config.h>

namespace oz {

    static texture<float4, 2> texSRC;

    template <bool tangential, bool src_linear>
    __global__ void oabf_filter( gpu_plm2<float3> dst, const gpu_plm2<float2> tf,
                                 float sigma_d, float sigma_r, float precision)
    {
        const int ix = blockDim.x * blockIdx.x + threadIdx.x;
        const int iy = blockDim.y * blockIdx.y + threadIdx.y;
        if (ix >= dst.w || iy >= dst.h) return;

        float2 t = tf(ix, iy);
        if (!tangential) t = make_float2(t.y, -t.x);

        float twoSigmaD2 = 2 * sigma_d * sigma_d;
        float twoSigmaR2 = 2 * sigma_r * sigma_r;
        int halfWidth = int(ceilf(precision * sigma_d));

        float2 tabs = fabs(t);
        float ds = (tabs.x > tabs.y)? 1.0f / tabs.x : 1.0f / tabs.y;

        float2 uv = make_float2(0.5f + ix, 0.5f + iy);
        float3 c0 = make_float3(tex2D(texSRC, uv.x, uv.y));
        float3 sum = c0;

        float norm = 1;
        float sign = -1;
        do {
            for (float d = ds; d <= halfWidth; d += ds) {
                float2 p = uv + d * t * sign;

                float kd = __expf( -dot(d,d) / twoSigmaD2 );
                if (src_linear) {
                    float3 c = make_float3(tex2D(texSRC, p.x, p.y));
                    float kr = __expf( -squared(c - c0) / twoSigmaR2 );
                    sum += kd * kr * c;
                    norm += kd * kr;
                } else {
                    p -= make_float2(0.5f, 0.5f);

                    float3 c1, c2;
                    float f;
                    if (tabs.x < tabs.y) {
                        float2 q = make_float2(floorf(p.x), p.y);
                        c1 = make_float3(tex2D(texSRC, q.x, q.y));
                        c2 = make_float3(tex2D(texSRC, q.x + 1, q.y));
                        f = p.x - q.x;
                    } else {
                        float2 q = make_float2(p.x, floorf(p.y));
                        c1 = make_float3(tex2D(texSRC, q.x, q.y));
                        c2 = make_float3(tex2D(texSRC, q.x, q.y + 1));
                        f = p.y - q.y;
                    }

                    float kr1 = (1 -  f) * __expf( -squared(c1 - c0) / twoSigmaR2 );
                    float kr2 = f * __expf( -squared(c2 - c0) / twoSigmaR2 );

                    sum += kd * (kr1 * c1 + kr2 * c2);
                    norm += kd * (kr1 + kr2);
                }
            }
            sign *= -1;
        } while (sign > 0);
        dst.write(ix, iy, sum / norm);
    }


    gpu_image oabf2( const gpu_image& src, const gpu_image& tf, float sigma_dg, float sigma_rg,
                     float sigma_dt, float sigma_rt, bool src_linear, float precision )
    {
        gpu_image dst0;
        if (sigma_dg <= 0) dst0 =  src;
        else {
            dst0 = gpu_image(src.size(), FMT_FLOAT3);
            gpu_binder<float3> src_(texSRC, src, src_linear? cudaFilterModeLinear : cudaFilterModePoint);
            launch_config cfg(dst0);
            if (src_linear)
                oabf_filter<false,true><<<cfg.blocks(), cfg.threads()>>>(dst0, tf, sigma_dg, sigma_rg, precision);
            else
                oabf_filter<false,false><<<cfg.blocks(), cfg.threads()>>>(dst0, tf, sigma_dg, sigma_rg, precision);
        }

        gpu_image dst1;
        if (sigma_dt <= 0) dst1 = dst0;
        else {
            dst1 = gpu_image(src.size(), FMT_FLOAT3);
            gpu_binder<float3> src_(texSRC, dst0, src_linear? cudaFilterModeLinear : cudaFilterModePoint);
            launch_config cfg(dst1);
            if (src_linear)
                oabf_filter<true,true><<<cfg.blocks(), cfg.threads()>>>(dst1, tf, sigma_dt, sigma_rt, precision);
            else
                oabf_filter<true,false><<<cfg.blocks(), cfg.threads()>>>(dst1, tf, sigma_dt, sigma_rt, precision);
        }

        return dst1;
    }

}

