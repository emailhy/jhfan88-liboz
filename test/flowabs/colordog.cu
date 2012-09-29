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
#include <oz/gpu_image.h>
#include <oz/gpu_binder.h>
#include <oz/launch_config.h>
#include <oz/gpu_plm2.h>
using namespace oz;

/*
__global__ void imp_mag_diff( gpu_plm2<float> dst, const gpu_plm2<float> src0, const gpu_plm2<float> src1 ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    dst(ix, iy) = fmax( 0.0f, src0(ix, iy) - src1(ix, iy) );
}


gpu_image<float> mag_diff( const gpu_image<float>& src0, const gpu_image<float>& src1 ) {
    gpu_image<float> dst(src0.size());
    imp_mag_diff<<<dst.blocks(), dst.threads()>>>(dst, src0, src1);
    GPU_CHECK_ERROR();
    return dst;
}
*/


static texture<float4, 2, cudaReadModeElementType> texSRC4;


/*
static __device__ float kstep(float x, float K, float B1, float B2) {
    if (x < B1) return K;
    if (x > B2) return 0;
    return K - (x - B1) / (B2 - B1);
}
*/


__global__ void imp_color_gdog( gpu_plm2<float3> dst, const gpu_plm2<float4> tfab,
                                float sigma_e, float sigma_r, float precision, float tau )
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float4 t = tfab(ix, iy);
    float2 n = make_float2(t.y, -t.x);
    float2 nabs = fabs(n);
    float ds = 1.0f / ((nabs.x > nabs.y)? nabs.x : nabs.y);

    float twoSigmaE2 = 2 * sigma_e * sigma_e;
    float twoSigmaR2 = 2 * sigma_r * sigma_r;
    float halfWidth = precision * sigma_r;

    float3 c0 = make_float3(tex2D(texSRC4, ix, iy));
    float3 sumE = c0;
    float3 sumR = sumE;
    float2 norm = make_float2(1, 1);

    for( float d = ds; d <= halfWidth; d += ds ) {
        float kE = __expf( -d * d / twoSigmaE2 );
        float kR = __expf( -d * d / twoSigmaR2 );

        float2 o = d*n;
        float3 c = make_float3(tex2D( texSRC4, 0.5f + ix - o.x, 0.5f + iy - o.y)) +
                   make_float3(tex2D( texSRC4, 0.5f + ix + o.x, 0.5f + iy + o.y));
        sumE += kE * c;
        sumR += kR * c;
        norm += 2 * make_float2(kE, kR);
    }
    sumE /= norm.x;
    sumR /= norm.y;

    float3 hp = sumE - sumR;
    dst.write(ix, iy, hp);
}


gpu_image color_gdog( const gpu_image& src, const gpu_image& tfab,
                      float sigma_e, float sigma_r, float precision, float tau )
{
    gpu_image dst(src.size(), FMT_FLOAT3);
    gpu_binder<float3> src_(texSRC4, src);
    launch_config cfg(dst);
    imp_color_gdog<<<cfg.blocks(), cfg.threads()>>>(dst, tfab, sigma_e, sigma_r, precision, tau);
    OZ_CUDA_ERROR_CHECK();
    return dst;
}


__global__ void imp_chroma_sharp( gpu_plm2<float> dst, const gpu_plm2<float> L, const gpu_plm2<float3> HP,
                                  float K, float B1, float B2 )
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float l = L(ix, iy);
    float3 hp = HP(ix, iy);
    float chp = sqrtf( hp.x*hp.x + hp.y*hp.y + hp.z*hp.z );

    dst.write(ix, iy, sign(hp.x) * chp);
}


gpu_image chroma_sharp( const gpu_image& L, const gpu_image& hp, float K, float B1, float B2) {
    gpu_image dst(L.size(), FMT_FLOAT);
    launch_config cfg(dst);
    imp_chroma_sharp<<<cfg.blocks(), cfg.threads()>>>(dst, L, hp, K, B1, B2);
    OZ_CUDA_ERROR_CHECK();
    return dst;
}

