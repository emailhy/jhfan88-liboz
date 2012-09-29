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
#include <oz/colormap.h>
#include <oz/colormap_util.h>
#include <oz/gpu_plm2.h>
#include <oz/launch_config.h>
using namespace oz;


__device__ inline float diff_angle(float x, float y, float a) {
    if (y < 0) {
        y = -y;
        x = -x;
    }
    float a0 = atan2f(-x, y);

    float e0 = a0 - a;
    float e1 = a0 - a - CUDART_PI_F;
    float e2 = a0 - a + CUDART_PI_F;

    float e = e0;
    if (fabs(e1) < fabs(e)) e = e1;
    if (fabs(e2) < fabs(e)) e = e2;

    return e * 180 / CUDART_PI_F;
}


__global__ void imp_diff_angle( gpu_plm2<float> dst, const gpu_plm2<float> angle) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float x = ix - 0.5f * dst.w + 0.5f;
    float y = iy - 0.5f * dst.h + 0.5f;

    if (sqrtf(x*x + y*y) < 0.475f*dst.w) {
        float a0 = atan2f(-x, y);
        float a1 = angle(ix, iy);
        float e;
        //if (x < dst.w /2)
        //    e = acos( cos(2*a0 - 2*a1) ) / 2 / CUDART_PIO2_F * 180.0f;
        //else
        e = diff_angle(x, y, a1);
        dst.write(ix, iy, e);
    } else {
        dst.write(ix ,iy, 0);
    }
}


gpu_image testring_diff_angle( const gpu_image& angle ) {
    gpu_image dst( angle.size(), FMT_FLOAT );
    launch_config cfg(dst);
    imp_diff_angle<<<cfg.blocks(), cfg.threads()>>>(dst, angle);
    OZ_CUDA_ERROR_CHECK();
    return dst;
}


__global__ void imp_xy_angle( gpu_plm2<float> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float x = ix - 0.5f * dst.w + 0.5f;
    float y = iy - 0.5f * dst.h + 0.5f;

    if (sqrtf(x*x + y*y) < 0.5f*dst.w) {
        if (y < 0) {
            y = -y;
            x = -x;
        }
        float a0 = atan2f(-x, y);
        dst.write(ix, iy, a0);
    } else {
        dst.write(ix ,iy, 0);
    }
}


gpu_image testring_xy_angle( int w, int h ) {
    gpu_image dst( w, h, FMT_FLOAT );
    launch_config cfg(dst);
    imp_xy_angle<<<cfg.blocks(), cfg.threads()>>>(dst);
    OZ_CUDA_ERROR_CHECK();
    return dst;
}


__global__ void imp_jet( gpu_plm2<float3> dst, const gpu_plm2<float> diff, float scale ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float x = ix - 0.5f * dst.w + 0.5f;
    float y = iy - 0.5f * dst.h + 0.5f;

    if (sqrtf(x*x + y*y) < 0.475f*dst.w) {
        dst.write(ix, iy, colormap_jet( diff(ix, iy) / scale ));
    } else {
        dst.write(ix ,iy, make_float3(0.5));
    }
}


gpu_image testring_jet( const gpu_image& diff, float scale ) {
    gpu_image dst( diff.size(), FMT_FLOAT3 );
    launch_config cfg(dst);
    imp_jet<<<cfg.blocks(), cfg.threads()>>>(dst, diff, scale);
    OZ_CUDA_ERROR_CHECK();
    return dst;
}


/*
__global__ void imp_8u( plm2<float3> dst, const plm2<float3> src ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float3 g = clamp(src(ix, iy), -1, 1);
    g *= 127;
    g += make_float3(0.5f)
    uchar3 c =
    dst.write(ix, iy, g);
}


gpu_image testring_8u( const gpu_image& src ) {
    gpu_image dst( src.size(), FMT_FLOAT3 );
    launch_config cfg(dst);
    imp_8u<<<cfg.blocks(), cfg.threads()>>>(dst, src);
    OZ_CUDA_ERROR_CHECK();
    return dst;
}
*/