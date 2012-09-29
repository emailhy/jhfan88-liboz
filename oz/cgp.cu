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
#include <oz/cgp.h>
#include <oz/st.h>
#include <oz/st_util.h>
#include <oz/gpu_plm2.h>
#include <oz/gpu_binder.h>
#include <oz/launch_config.h>


static texture<float, 2, cudaReadModeElementType> texSRC1;
static texture<float4, 2, cudaReadModeElementType> texSRC4;
static texture<float4, 2, cudaReadModeElementType> texST;

namespace oz {
    struct cgp_lic_t {
        size_t w;
        size_t h;
        float twoSigma2;
        float halfWidth;
        float2 p0;
        float2 v0;
        float3 c;
        float z;
        float sum;
        float2 v;
        float2 p;
        float stop;

        __device__ cgp_lic_t(float2 _p0, size_t _w, size_t _h, float sigma, float mstop) {
            w = _w;
            h = _h;
            p0 = _p0;
            c = make_float3(0);
            z = tex2D(texSRC1, p0.x, p0.y);
            v0 = st2tangent(make_float3(tex2D(texST, p0.x, p0.y)));
            sum = 1;
            twoSigma2 = 2 * sigma * sigma;
            halfWidth = 2 * sigma;
            stop = mstop;
        }

        __device__ void cgp( float sign ) {
            v = v0 * sign;
            p = p0 + v;
            float r = 1;
            while ((r < halfWidth) && (p.x >= 0) && (p.x < w) && (p.y >= 0) && (p.y < h))  {
                float k = __expf(-r * r / twoSigma2);
                float tempz = tex2D(texSRC1, p.x, p.y);
                if (tempz > z) {
                    z = tempz;
                }

                float2 t = st2tangent(make_float3(tex2D(texST, p.x, p.y)));
                float vt = dot(v, t);
                if (fabs(vt) <= stop) break;
                if (vt < 0) t = -t;

                v = t;
                p += t;
                r += 1;
            }
        }

        __device__ void cross_cgp( float sign ) {
            v = v0 * sign;
            p = p0 + v;
            float r = 1;
            while ((r < halfWidth) && (p.x >= 0) && (p.x < w) && (p.y >= 0) && (p.y < h))  {
                float k = __expf(-r * r / twoSigma2);
                float tempz = tex2D(texSRC1, p.x, p.y);
                if (tempz > z) {
                    c = make_float3(tex2D(texSRC4, p.x, p.y));
                    z = tempz;
                }

                float2 t = st2tangent(make_float3(tex2D(texST, p.x, p.y)));
                float vt = dot(v, t);
                if (fabs(vt) <= stop) break;
                if (vt < 0) t = -t;

                v = t;
                p += t;
                r += 1;
            }
        }
    };


    __global__ void imp_stgauss_euler_cgp( gpu_plm2<float> dst, float sigma, float cos_max ) {
        const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
        const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
        if(ix >= dst.w || iy >= dst.h)
            return;

        float2 uv = make_float2(ix + 0.5f, iy + 0.5f);
        cgp_lic_t L(uv, dst.w, dst.h, sigma, cos_max);
        L.cgp(+1);
        L.cgp(-1);
        dst.write(ix, iy, L.z);
    }


    __global__ void imp_stgauss_euler_cross_cgp( gpu_plm2<float3> dst, float sigma, float cos_max ) {
        const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
        const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
        if(ix >= dst.w || iy >= dst.h)
            return;

        float2 uv = make_float2(ix + 0.5f, iy + 0.5f);
        cgp_lic_t L(uv, dst.w, dst.h, sigma, cos_max);
        L.c = make_float3(tex2D(texSRC4, uv.x, uv.y));
        L.cross_cgp(+1);
        L.cross_cgp(-1);
        dst.write(ix, iy, L.c);
    }
}


oz::gpu_image oz::cgp_filter( const gpu_image& src, const gpu_image& st,
                              float sigma, float max_angle )
{
    if (sigma <= 0) return src;
    if (src.size() != st.size()) OZ_INVALID_SIZE();

    gpu_image dst(src.size(), FMT_FLOAT);
    gpu_binder<float> src_(texSRC1, src);
    gpu_binder<float3> st_(texST, st, cudaFilterModeLinear);

    float cos_max = cos(max_angle * CUDART_PI_F/180.0f);
    launch_config cfg(dst);
    imp_stgauss_euler_cgp<<<cfg.blocks(), cfg.threads()>>>( dst, sigma, cos_max );
    OZ_CUDA_ERROR_CHECK();
    return dst;
}


oz::gpu_image oz::cgp_cross_filter( const gpu_image& src, const gpu_image& noise,
                                    const gpu_image& st, float sigma, float max_angle )
{
    if (sigma <= 0) return src;
    if (src.size() == st.size()) OZ_INVALID_SIZE();
    gpu_image dst(src.size(), FMT_FLOAT3);
    gpu_binder<float3> src_(texSRC4, src);
    gpu_binder<float> noise_(texSRC1, noise);
    gpu_binder<float3> st_(texST, st, cudaFilterModeLinear);

    float cos_max = cos(max_angle * CUDART_PI_F/180.0f);
    launch_config cfg(dst);
    imp_stgauss_euler_cross_cgp<<<cfg.blocks(), cfg.threads()>>>( dst, sigma, cos_max );
    OZ_CUDA_ERROR_CHECK();
    return dst;
}
