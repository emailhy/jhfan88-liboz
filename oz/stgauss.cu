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
#include <oz/stgauss.h>
#include <oz/st.h>
#include <oz/st_util.h>
#include <oz/gpu_plm2.h>
#include <oz/gpu_binder.h>
#include <oz/launch_config.h>
using namespace oz;


static texture<float, 2, cudaReadModeElementType> texSRC1;
static texture<float4, 2, cudaReadModeElementType> texSRC4;
static texture<float4, 2, cudaReadModeElementType> texST4;

template<typename T> __device__ T texSRC(float x, float y);
template<> inline __device__ float texSRC(float x, float y) { return tex2D(texSRC1, x, y); }
template<> inline __device__ float3 texSRC(float x, float y) { return make_float3(tex2D(texSRC4, x, y)); }
inline __device__ float3 texST(float x, float y) { return make_float3(tex2D(texST4, x, y)); }


template <typename T>
struct st_lic_t {
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
    float stop;

    __device__ st_lic_t(float2 _p0, size_t _w, size_t _h, float sigma, float mstop) {
        w = _w;
        h = _h;
        p0 = _p0;
        c = texSRC<T>(p0.x, p0.y);
        v0 = st2tangent(texST(p0.x, p0.y));
        sum = 1;
        twoSigma2 = 2 * sigma * sigma;
        halfWidth = 2 * sigma;
        stop = mstop;
    }

    __device__ void smooth( float sign ) {
        v = v0 * sign;
        p = p0 + v;
        float r = 1;
        while ((r < halfWidth) && (p.x >= 0) && (p.x < w) && (p.y >= 0) && (p.y < h))  {
            float k = __expf(-r * r / twoSigma2);
            c += texSRC<T>(p.x, p.y) * k;
            sum += k;

            float2 t = st2tangent(texST(p.x, p.y));
            float vt = dot(v, t);
            if (fabs(vt) <= stop) break;
            if (vt < 0) t = -t;

            v = t;
            p += t;
            r += 1;
        }
    }

    __device__ void smooth_cgp( float sign ) {
        v = v0 * sign;
        p = p0 + v;
        float r = 1;
        while ((r < halfWidth) && (p.x >= 0) && (p.x < w) && (p.y >= 0) && (p.y < h))  {
            float k = __expf(-r * r / twoSigma2);
            c = fmax(c, texSRC<T>(p.x, p.y));

            float2 t = st2tangent(texST(p.x, p.y));
            float vt = dot(v, t);
            if (fabs(vt) <= stop) break;
            if (vt < 0) t = -t;

            v = t;
            p += t;
            r += 1;
        }
    }

    __device__ void smooth_rungekutta( float sign ) {
        v = v0 * sign;

        float2 t = st2tangent(texST(p0.x + 0.5f * v.x, p0.y + 0.5f * v.y));
        float vt = dot(v, t);
        if (vt < 0) t = -t;
        v = t;
        p = p0 + v;

        float r = 1;
        while ((r < halfWidth) && (p.x >= 0) && (p.x < w) && (p.y >= 0) && (p.y < h))  {
            float k = __expf(-r * r / twoSigma2);
            c += texSRC<T>(p.x, p.y) * k;
            sum += k;

            t = st2tangent(texST(p.x, p.y));
            vt = dot(v, t);
            if (fabs(vt) <= stop) break;
            if (vt < 0) t = -t;

            t = st2tangent(texST(p.x + 0.5f * t.x, p.y + 0.5f * t.y));
            vt = dot(v, t);
            if (fabs(vt) <= stop) break;
            if (vt < 0) t = -t;

            v = t;
            p += t;
            r += 1;
        }
    }
};


template<typename T>
__global__ void imp_stgauss_euler( oz::gpu_plm2<T> dst, float sigma, float cos_max, bool adaptive ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    if (adaptive) {
        float A = st2A(texST(ix+0.5f, iy+0.5f));
        sigma *= 0.25f * (1 + A)*(1 + A);
    }

    float2 uv = make_float2(ix + 0.5f, iy + 0.5f);
    st_lic_t<T> L(uv, dst.w, dst.h, sigma, cos_max);
    L.smooth(+1);
    L.smooth(-1);
    dst.write(ix, iy, L.c / L.sum);
}


template<typename T>
__global__ void imp_stgauss_rungekutta( oz::gpu_plm2<T> dst, float sigma, float cos_max, bool adaptive ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    if (adaptive) {
        float A = st2A(texST(ix+0.5f, iy+0.5f));
        sigma *= 0.25f * (1 + A)*(1 + A);
    }

    float2 uv = make_float2(ix + 0.5f, iy + 0.5f);
    st_lic_t<T> L(uv, dst.w, dst.h, sigma, cos_max);
    L.smooth_rungekutta(+1);
    L.smooth_rungekutta(-1);
    dst.write(ix, iy, L.c / L.sum);
}


oz::gpu_image oz::stgauss_filter( const gpu_image& src, const gpu_image& st,
                                  float sigma, float max_angle, bool adaptive,
                                  bool src_linear, bool st_linear, int order )
{
    if (sigma <= 0) return src;
    if (src.size() != st.size()) OZ_INVALID_SIZE();
    float cos_max = cos(max_angle * CUDART_PI_F/180.0f);
    switch (src.format()) {
        case FMT_FLOAT:
            {
                gpu_image dst(src.size(), FMT_FLOAT);
                gpu_binder<float> src_(texSRC1, src, src_linear? cudaFilterModeLinear : cudaFilterModePoint);
                gpu_binder<float3> st_(texST4, st, st_linear? cudaFilterModeLinear : cudaFilterModePoint);
                launch_config cfg(dst);
                if (order == 1) {
                    imp_stgauss_euler<float><<<cfg.blocks(), cfg.threads()>>>( dst, sigma, cos_max, adaptive);
                } else {
                    imp_stgauss_rungekutta<float><<<cfg.blocks(), cfg.threads()>>>( dst, sigma, cos_max, adaptive);
                }
                OZ_CUDA_ERROR_CHECK();
                return dst;
            }

        case FMT_FLOAT3:
            {
                gpu_image dst(src.size(), FMT_FLOAT3);
                gpu_binder<float3> src_(texSRC4, src, src_linear? cudaFilterModeLinear : cudaFilterModePoint);
                gpu_binder<float3> st_(texST4, st, st_linear? cudaFilterModeLinear : cudaFilterModePoint);
                launch_config cfg(dst);
                if (order == 1) {
                    imp_stgauss_euler<float3><<<cfg.blocks(), cfg.threads()>>>( dst, sigma, cos_max, adaptive);
                } else {
                    imp_stgauss_rungekutta<float3><<<cfg.blocks(), cfg.threads()>>>( dst, sigma, cos_max, adaptive);
                }
                OZ_CUDA_ERROR_CHECK();
                return dst;
            }

        default:
            OZ_INVALID_FORMAT();
    }
}
