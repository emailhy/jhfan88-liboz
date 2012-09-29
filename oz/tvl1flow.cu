//
// TV-L1 optical flow:
// Antonin Chambolle and Thomas Pock, A first-order primal-dual
// algorithm with applications to imaging, Technical Report, 2010
//
// by Jan Eric Kyprianidis <www.kyprianidis.com>
// based on a MATLAB implementation by Thomas Pock 
// Copyright 2011 Adobe Systems Incorporated 
// 
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
#include <oz/tvl1flow.h>
#include <oz/gpu_binder.h>
#include <oz/resample.h>
#include <oz/resize.h>
#include <oz/peakfilt.h>
#include <oz/make.h>
#include <oz/shuffle.h>
#include <oz/gpu_plm2.h>
#include <oz/launch_config.h>
#include <vector>
using namespace oz;


static texture<float, 2, cudaReadModeElementType> texI2;


__global__ void imp_tvl1_warp( gpu_plm2<float4> I_xytq, // I_x, I_y, I_t, I_
                               const gpu_plm2<float> I1,
                               const gpu_plm2<float> u, 
                               const gpu_plm2<float> v,
                               float gamma ) 
{
    const unsigned ix = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix >= I1.w || iy >= I1.h)
        return;

    float idxx = ix + 0.5f + u(ix, iy);
    float idxm = idxx - 0.5f;
    float idxp = idxx + 0.5f;

    float idyy = iy + 0.5f + v(ix, iy);
    float idym = idyy - 0.5f;
    float idyp = idyy + 0.5f;

    if ((ix != 0) && (ix != I1.w-1) && (iy != 0) && (iy != I1.h-1)){
        float i2_warped = tex2D(texI2, idxx, idyy);
        float i_x = tex2D(texI2, idxp, idyy) - tex2D(texI2, idxm, idyy);
        float i_y = tex2D(texI2, idxx, idyp) - tex2D(texI2, idxx, idym);
        float i_t = i2_warped - I1(ix,iy);
        float i_grad_sqr = fmaxf(1e-7, i_x*i_x + i_y*i_y + gamma*gamma);
        I_xytq.write(ix,iy, make_float4(i_x, i_y, i_t, i_grad_sqr));
    } else {
        I_xytq.write(ix,iy, make_float4(0, 0, 0, gamma*gamma));
    }
}                       


static void tvl1_warp( gpu_image& I_xytq, const gpu_image& I1, const gpu_image& I2, 
                       const gpu_image& u, const gpu_image& v, float gamma )
{
    launch_config cfg(I1);
    gpu_binder<float> I2_(texI2, I2, cudaFilterModeLinear);
    imp_tvl1_warp<<<cfg.blocks(), cfg.threads()>>>(I_xytq, I1, u, v, gamma);
    OZ_CUDA_ERROR_CHECK();
}


static texture<float4, 2, cudaReadModeElementType> texUVW_;


__global__ void imp_tvl1_update_dual( gpu_plm2<float4> p0, 
                                      gpu_plm2<float2> p1,
                                      float sigma )
{
    const unsigned ix = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned iy = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned iw = p0.w;
    const unsigned ih = p0.h;
    if (ix >= iw || iy >= ih)
        return;

    float3 tmp = make_float3(tex2D(texUVW_, ix, iy));
    float3 uvw_x = (ix + 1 == iw)? make_float3(0) : make_float3(tex2D(texUVW_, ix + 1, iy)) - tmp;
    float3 uvw_y = (iy + 1 == ih)? make_float3(0) : make_float3(tex2D(texUVW_, ix, iy + 1)) - tmp;

    float4 q0 =  p0(ix, iy); 
    float2 q1 =  p1(ix, iy);
    q0.x += sigma * uvw_x.x;
    q0.y += sigma * uvw_y.x;
    q0.z += sigma * uvw_x.y;
    q0.w += sigma * uvw_y.y;
    q1.x += sigma * uvw_x.z;
    q1.y += sigma * uvw_y.z;

    float r;
    r = fmaxf(1.0f, sqrtf( q0.x*q0.x + q0.y*q0.y + q0.z*q0.z + q0.w*q0.w));
    p0.write(ix, iy, q0 / r);

    r = fmaxf(1.0f, sqrtf(q1.x*q1.x + q1.y*q1.y));
    p1.write(ix, iy, q1 / r);
}



static void tvl1_update_dual( gpu_image& p0, gpu_image& p1, const gpu_image& uvw_, float sigma) {
    launch_config cfg(p0);
    gpu_binder<float3> uvw__(texUVW_, uvw_);
    imp_tvl1_update_dual<<<cfg.blocks(), cfg.threads()>>>(p0, p1, sigma);
    OZ_CUDA_ERROR_CHECK();
}


static texture<float4, 2, cudaReadModeElementType> texP0_;
static texture<float2, 2, cudaReadModeElementType> texP1_;


__global__ void imp_tvl1_update_primal( gpu_plm2<float3> UVW, 
                                        gpu_plm2<float3> UVW_,
                                        gpu_plm2<float2> UV0,
                                        const gpu_plm2<float4> I_xytq, 
                                        float tau,
                                        float lambda,
                                        float gamma ) 
{
    const unsigned ix = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned iy = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned iw = UVW.w;
    const unsigned ih = UVW.h;
    if (ix >= iw || iy >= ih)
        return;

    float4 i_xytq = I_xytq(ix, iy);
    float i_x = i_xytq.x;
    float i_y = i_xytq.y;
    float i_t = i_xytq.z;
    float i_grad_sqr = i_xytq.w; //fmaxf(1e-7, i_x*i_x + i_y*i_y + gamma*gamma);

    float4 p0ii = tex2D(texP0_, ix, iy);
    float2 p1ii = tex2D(texP1_, ix, iy);
    float div_u, div_v, div_w;
    if (ix == 0) {
        div_u = p0ii.x;
        div_v = p0ii.z;
        div_w = p1ii.x;
    } else if (ix == iw-1) {
        float4 p0 = -1*tex2D(texP0_, ix-1, iy);
        float2 p1 = -1*tex2D(texP1_, ix-1, iy);
        div_u = p0.x;
        div_v = p0.z;
        div_w = p1.x;
    } else {
        float4 p0 = tex2D(texP0_, ix-1, iy);
        float2 p1 = tex2D(texP1_, ix-1, iy);
        div_u = p0ii.x - p0.x;
        div_v = p0ii.z - p0.z;
        div_w = p1ii.x - p1.x;
    }

    if (iy == 0) {
        div_u += p0ii.y;
        div_v += p0ii.w;
        div_w += p1ii.y;
    } else if (iy == ih-1) {
        float4 p0 = -1*tex2D(texP0_, ix, iy-1);
        float2 p1 = -1*tex2D(texP1_, ix, iy-1);
        div_u += p0.y;
        div_v += p0.w;
        div_w += p1.y;
    } else {
        float4 p0 = tex2D(texP0_, ix, iy-1);
        float2 p1 = tex2D(texP1_, ix, iy-1);
        div_u += p0ii.y - p0.y;
        div_v += p0ii.w - p0.w;
        div_w += p1ii.y - p1.y;
    }

    float3 uvw_ = UVW(ix, iy);
    float u_ = uvw_.x;
    float v_ = uvw_.y;
    float w_ = uvw_.z;
    float2 uv0 = UV0(ix,iy);

    float u = u_ + tau * div_u;
    float v = v_ + tau * div_v;
    float w = w_ + tau * div_w;

    float rho = i_t + (u-uv0.x)*i_x + (v-uv0.y)*i_y + gamma*w;

    if (rho < -tau*lambda*i_grad_sqr) {
        u += tau * lambda * i_x;
        v += tau * lambda * i_y;
        w += tau * lambda * gamma;
    } else if (rho > tau*lambda*i_grad_sqr) {
        u -= tau * lambda * i_x;
        v -= tau * lambda * i_y;
        w -= tau * lambda * gamma;
    } else {
        u -= rho * i_x / i_grad_sqr;
        v -= rho * i_y / i_grad_sqr;
        w -= rho * gamma / i_grad_sqr;
    }

    UVW.write(ix,iy, make_float3(u, v, w));
    UVW_.write(ix,iy, make_float3( 2*u - u_, 2*v - v_, 2*w - w_));
}


static void tvl1_update_primal( const gpu_image& p0, 
                                const gpu_image& p1,
                                gpu_image& uvw, 
                                gpu_image& uvw_, 
                                const gpu_image& uv0, 
                                const gpu_image& I_xytq, 
                                float tau, 
                                float lambda,
                                float gamma ) 
{
    launch_config cfg(p0);
    gpu_binder<float4> P0_(texP0_, p0, cudaFilterModePoint);
    gpu_binder<float2> P1_(texP1_, p1, cudaFilterModePoint);
    imp_tvl1_update_primal<<<cfg.blocks(), cfg.threads()>>>(uvw, uvw_, uv0, I_xytq, tau, lambda, gamma);
    OZ_CUDA_ERROR_CHECK();
}


gpu_image oz::tvl1flow( const gpu_image& src0, const gpu_image& src1,
                        float pyr_scale, int warps, int maxits, float lambda ) 
{ 
    std::vector<gpu_image> P0, P1;
    {
        gpu_image s0 = src0;
        gpu_image s1 = src1;
        for (;;) {
            P0.push_back(s0);
            P1.push_back(s1);
            int w = (int)(s0.w() * pyr_scale);
            int h = (int)(s0.h() * pyr_scale);
            if ((w <= 16) || (h <= 16)) break;
            s0 = resample(s0, w, h, RESAMPLE_CUBIC);
            s1 = resample(s1, w, h, RESAMPLE_CUBIC);
        }
    }

    gpu_image p0;
    gpu_image p1;
    gpu_image uvw;

    for (int level = (int)P0.size()-1; level >= 0; --level) {
        int N = P0[level].w();
        int M = P0[level].h();

        if (level == P0.size()-1) {
            uvw = gpu_image(N, M, FMT_FLOAT3);
            p0 = gpu_image(N, M, FMT_FLOAT4);
            p1 = gpu_image(N, M, FMT_FLOAT2);
            uvw.clear();
            p0.clear();
            p1.clear();
        } else {
            float su = (float)P0[level+1].w() / P0[level].w();
            float sv = (float)P0[level+1].h() / P0[level].h();

            uvw = resize(uvw, N, M, RESIZE_NEAREST);
            uvw = uvw * make_float3(su, sv, 1);

            p0 = resize(p0, N, M, RESIZE_NEAREST);
            p1 = resize(p1, N, M, RESIZE_NEAREST);
        }

        gpu_image I1 = P0[level];
        gpu_image I2 = P1[level];
        {
            float L = sqrtf(8);
            float tau = 1.0f/L;
            float sigma = 1.0F/L;
            float gamma = 0.02f;

            gpu_image uvw_ = uvw.clone();

            for (int j = 0; j < warps; ++j) {

                gpu_image uv0 = shuffle(uvw, 0, 1);
                gpu_image I_xytq(N, M, FMT_FLOAT4);
                tvl1_warp(I_xytq, I1, I2, shuffle(uv0, 0), shuffle(uv0, 1), gamma);

                for (int k = 0; k < maxits; ++k) {
                    tvl1_update_dual(p0, p1, uvw_, sigma);
                    tvl1_update_primal(p0, p1, uvw, uvw_, uv0, I_xytq, tau, lambda, gamma );
                }

                {
                    gpu_image u = shuffle(uvw, 0);
                    gpu_image v = shuffle(uvw, 1);
                    gpu_image w = shuffle(uvw, 2);
                    u = peakfilt_3x3(u);
                    v = peakfilt_3x3(v);
                    uvw = make(u,v,w);
                }
            }
        }
    }
    
   return shuffle(uvw, 0, 1);
}
