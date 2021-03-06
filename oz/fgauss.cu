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
#include <oz/fgauss.h>
#include <oz/gpu_plm2.h>
#include <oz/gpu_binder.h>
#include <oz/launch_config.h>


static texture<float, 2, cudaReadModeElementType> texSRC1;
static texture<float4, 2, cudaReadModeElementType> texSRC4;
static texture<float2, 2, cudaReadModeElementType> texTM;

template<typename T> __device__ T texSRC(float x, float y);
template<> inline __device__ float texSRC(float x, float y) { return tex2D(texSRC1, x, y); }
template<> inline __device__ float3 texSRC(float x, float y) { return make_float3(tex2D(texSRC4, x, y)); }


template <typename T>
struct tfm_lic_t {
    size_t w;
    size_t h;
    float twoSigma2;
    float halfWidth;
    float2 p0;
    float2 v0;
    T c;
    float sum;
    float2 v;
    float2 p;
    float step;

    __device__ tfm_lic_t( float2 _p0, size_t _w, size_t _h,
                          float sigma, float precision, float step_size)
    {
        w = _w;
        h = _h;
        p0 = _p0;
        c = texSRC<T>(p0.x, p0.y);
        v0 = tex2D(texTM, p0.x, p0.y);
        sum = 1;
        twoSigma2 = 2 * sigma * sigma;
        halfWidth = precision * sigma;
        step = step_size;
    }

    __device__ void smooth( float sign ) {
        v = v0 * sign;
        p = p0 + v;
        float r = 1;
        while ((r < halfWidth) && (p.x >= 0) && (p.x < w) && (p.y >= 0) && (p.y < h))  {
            float k = __expf(-r * r / twoSigma2);
            c += texSRC<T>(p.x, p.y) * k;
            sum += k;

            float2 t = step * tex2D(texTM, p.x, p.y);
            float vt = dot(v, t);
            if (vt == 0) break;
            if (vt < 0) t = -t;

            v = t;
            p += t;
            r += 1;
        }
    }
};


template<typename T>
__global__ void imp_fgauss_filter( oz::gpu_plm2<T> dst, float sigma, float precision, float step_size ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float2 uv = make_float2(ix + 0.5f, iy + 0.5f);
    tfm_lic_t<T> L(uv, dst.w, dst.h, sigma, precision, step_size);
    L.smooth(+1);
    L.smooth(-1);
    dst.write(ix, iy,  L.c / L.sum);
}


oz::gpu_image oz::fgauss_filter( const gpu_image& src, const gpu_image& tm,
                                float sigma, float precision, float step_size )
{
    if (sigma <= 0) return src;
    if (src.size() != tm.size()) OZ_INVALID_SIZE();
    switch (src.format()) {
        case FMT_FLOAT:
            {
                gpu_image dst(src.size(), FMT_FLOAT);
                gpu_binder<float> src_(texSRC1, src, cudaFilterModeLinear);
                gpu_binder<float2> tm_(texTM, tm);
                launch_config cfg(dst);
                imp_fgauss_filter<float><<<cfg.blocks(), cfg.threads()>>>( dst, sigma, precision, step_size );
                OZ_CUDA_ERROR_CHECK();
                return dst;
            }

        case FMT_FLOAT3:
            {
                gpu_image dst(src.size(), FMT_FLOAT3);
                gpu_binder<float3> src_(texSRC4, src, cudaFilterModeLinear);
                gpu_binder<float2> tm_(texTM, tm);
                launch_config cfg(dst);
                imp_fgauss_filter<float3><<<cfg.blocks(), cfg.threads()>>>( dst, sigma, precision, step_size );
                OZ_CUDA_ERROR_CHECK();
                return dst;
            }

        default:
            OZ_INVALID_FORMAT();
    }
}
