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
#include <oz/gpu_plm2.h>
#include <oz/st.h>
#include <oz/st_util.h>
#include <oz/launch_config.h>
using namespace oz;


texture<float4, 2, cudaReadModeElementType> texSRC;
texture<float, 2, cudaReadModeElementType> texKRNL;


__global__ void imp_anisotropic_smooth( const gpu_plm2<float4> R, gpu_plm2<float4> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float4 r = R(ix, iy);
    float cos_phi = cosf(r.z);
    float sin_phi = sinf(r.z);
    float3 uuT = make_float3(cos_phi*cos_phi, sin_phi*sin_phi, cos_phi*sin_phi);
    float3 vvT = make_float3(sin_phi*sin_phi, cos_phi*cos_phi, -cos_phi*sin_phi);
    float3 S = uuT / r.x + vvT / r.y;

    int max_x = int(sqrtf(r.x*r.x * cos_phi*cos_phi +
                          r.y*r.y * sin_phi*sin_phi));
    int max_y = int(sqrtf(r.x*r.x * sin_phi*sin_phi +
                          r.y*r.y * cos_phi*cos_phi));

    float4 sum = make_float4(0);
    float norm = 0;
    for (int j = -max_y; j <= max_y; ++j) {
        for (int i = -max_x; i <= max_x; ++i) {
            float kernel = __expf(-(i*i*S.x + 2*i*j*S.z + j*j*S.y));
            float4 c = tex2D(texSRC, ix + i, iy + j);
            sum += kernel * c;
            norm += kernel;
        }
    }
    sum /=  norm;

    dst.write(ix, iy, sum);
}


gpu_image gpu_anisotropic_smooth( const gpu_image& src, const gpu_image& R ) {
    gpu_image dst(src.size(), FMT_FLOAT4);
    gpu_binder<float4> src_(texSRC, src);
    launch_config cfg(dst);
    imp_anisotropic_smooth<<<cfg.blocks(), cfg.threads()>>>( R, dst );
    OZ_CUDA_ERROR_CHECK();
    return dst;
}


__global__ void imp_structure_tensor_adaptive_update( const gpu_plm2<float4> R, const gpu_plm2<float4> st,
                                                      gpu_plm2<float4> dst, float sigma_min, float sigma_max )
{
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float3 g = make_float3(st(ix, iy));
    float4 r = R(ix, iy);

    float lambda1 = 0.5f * (g.y + g.x +
        sqrtf(g.y*g.y - 2*g.x*g.y + g.x*g.x + 4*g.z*g.z));
    float lambda2 = 0.5f * (g.y + g.x -
        sqrtf(g.y*g.y - 2*g.x*g.y + g.x*g.x + 4*g.z*g.z));

    float phi = 0;
    if (lambda1 > 1e-5f) {
        phi = 0.5f * atan2(2 * g.z, g.x - g.y);

        //float l1 = lambda1 / (lambda1 + lambda2);
        //float l2 = lambda2 / (lambda1 + lambda2);

        //float a1 = l1 * sigma_min*sigma_min + (1 - l1) * sigma_max*sigma_max;
        //float a2 = l2 * sigma_min*sigma_min + (1 - l2) * sigma_max*sigma_max;

        float a2 = __powf(1 / (1 + lambda1 + lambda2), 0.5f) * sigma_max;
        float a1 = __powf(1 / (1 + lambda1 + lambda2), 2) * sigma_max;

        dst.write(ix, iy, make_float4(
            max(a1, 2.0f),
            max(a2, 2.0f),
            //clamp( (1+A) * r.x, 1.0f, 2*4.0f),
            //clamp( 1.0f / (1 + A) * r.y, 1.0f, 4.0f),
            //clamp(lambda1 / (lambda1 + lambda2) * r.x, 1.0f, 2*4.0f),
            //clamp(lambda2 / (lambda1 + lambda2) * r.y, 1.0f, 4.0f),
            phi,
            1
        ));
    } else {
        dst.write(ix, iy, make_float4(
            1,
            1,
            0,
            1
        ));
    }

    /*
    float2 t;
    if ((lambda1 > 1e-5)) {
        float phi = 0.5 * atan2(2 * g.z, g.x - g.y);
        t = make_float2(sinf(phi), -cosf(phi));
    } else {
        t = make_float2(0, 1);
        lambda1 = lambda2 = 0;
    }
    */
}


gpu_image gpu_st_adaptive_update( const gpu_image& R, const gpu_image& st,
                                                        float sigma_min, float sigma_max ) {
    gpu_image dst( R.size(), FMT_FLOAT4 );
    launch_config cfg(dst);
    imp_structure_tensor_adaptive_update<<<cfg.blocks(), cfg.threads()>>>( R, st, dst, sigma_min, sigma_max );
    OZ_CUDA_ERROR_CHECK();
    return dst;
}


__global__ void imp_structure_tensor_jacobi_step( gpu_plm2<float4> dst ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float4 c = tex2D(texSRC, ix, iy);
    float3 o;
    if (c.w > 0) {
        o = make_float3(c);
    } else {
        o = 0.25f * (
           make_float3(tex2D(texSRC, ix+1, iy)) +
           make_float3(tex2D(texSRC, ix-1, iy)) +
           make_float3(tex2D(texSRC, ix, iy+1)) +
           make_float3(tex2D(texSRC, ix, iy-1))
        );
    }

    dst.write(ix, iy, make_float4( o, c.w ));
}


gpu_image gpu_st_jacobi_step(const gpu_image& src) {
    gpu_image dst(src.size(), FMT_FLOAT4);
    gpu_binder<float4> src_(texSRC, src);
    launch_config cfg(dst);
    imp_structure_tensor_jacobi_step<<<cfg.blocks(), cfg.threads()>>>(dst);
    OZ_CUDA_ERROR_CHECK();
    return dst;
}


__global__ void imp_structure_tensor_relax_down( const gpu_plm2<float4> src, gpu_plm2<float4> dst) {
    int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float4 sum = make_float4(0);
    {
        float4 c = src(2*ix, 2*iy);
        if (c.w > 0) sum += make_float4(make_float3(c), 1);
    }

    if (2*ix+1 < src.w) {
        float4 c = src(2*ix+1, 2*iy);
        if (c.w > 0) sum += make_float4(make_float3(c), 1);
    }

    if (2*iy+1 < src.h) {
        float4 c = src(2*ix, 2*iy+1);
        if (c.w > 0) sum += make_float4(make_float3(c), 1);

        if (2*ix+1 < src.w) {
            float4 c = src(2*ix+1, 2*iy+1);
            if (c.w > 0) sum += make_float4(make_float3(c), 1);
        }
    }

    if (sum.w > 0) {
        dst.write(ix, iy, make_float4(make_float3(sum) / sum.w, 1));
    } else {
        dst.write(ix, iy, make_float4(0));
    }
}


gpu_image gpu_st_relax_down( const gpu_image& src ) {
    gpu_image dst((src.w()+1)/2, (src.h()+1)/2, FMT_FLOAT4);
    launch_config cfg(dst);
    imp_structure_tensor_relax_down<<<cfg.blocks(), cfg.threads()>>>(src, dst);
    OZ_CUDA_ERROR_CHECK();
    return dst;
}


__global__ void imp_structure_tensor_relax_up( const gpu_plm2<float4> src0, const gpu_plm2<float4> src1, gpu_plm2<float4> dst) {
    int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float4 c = src0(ix, iy);
    if (c.w == 0) {
        c = make_float4(make_float3(src1(ix/2, iy/2)), 0);
    }
    dst.write(ix, iy, c);
}


gpu_image gpu_st_relax_up( const gpu_image& src0, const gpu_image& src1 ) {
    gpu_image dst(src0.size(), FMT_FLOAT4);
    launch_config cfg(dst);
    imp_structure_tensor_relax_up<<<cfg.blocks(), cfg.threads()>>>(src0, src1, dst);
    OZ_CUDA_ERROR_CHECK();
    return dst;
}


gpu_image gpu_st_relax( const gpu_image& st ) {
    if ((st.w() <= 1) || (st.h() <= 1))
        return st;
    gpu_image tmp = gpu_st_relax(gpu_st_relax_down(st));
    return gpu_st_jacobi_step(gpu_st_relax_up(st, tmp));
}


__global__ void imp_smooth_3( const gpu_plm2<float4> src0, const gpu_plm2<float4> src1,
                              const gpu_plm2<float4> src2, gpu_plm2<float4> dst) {
    int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float3 c0 = make_float3(src0(ix, iy));
    float3 c1 = make_float3(src1(ix, iy));
    float3 c2 = make_float3(src2(ix, iy));

    dst.write(ix, iy, make_float4((c0 + 2*c1 + c2) / 4, 0));
}


gpu_image gpu_smooth_3( const gpu_image& src0, const gpu_image& src1, const gpu_image& src2 ) {
    gpu_image dst(src0.size(), FMT_FLOAT4);
    launch_config cfg(dst);
    imp_smooth_3<<<cfg.blocks(), cfg.threads()>>>(src0, src1, src2, dst);
    OZ_CUDA_ERROR_CHECK();
    return dst;
}


__global__ void imp_smooth_5( const gpu_plm2<float4> src0, const gpu_plm2<float4> src1, const gpu_plm2<float4> src2,
                              const gpu_plm2<float4> src3, const gpu_plm2<float4> src4, gpu_plm2<float4> dst) {
    int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float3 c0 = make_float3(src0(ix, iy));
    float3 c1 = make_float3(src1(ix, iy));
    float3 c2 = make_float3(src2(ix, iy));
    float3 c3 = make_float3(src2(ix, iy));
    float3 c4 = make_float3(src2(ix, iy));

    //dst(ix, iy) = make_float4((c0 + 4*c1 + 6*c2 + 4 * c3 + c4) / 16, 0);
    dst.write(ix, iy, make_float4((c0 + c1 + c2 + c3 + c4) / 5, 0));
}


gpu_image gpu_smooth_5( const gpu_image& src0, const gpu_image& src1, const gpu_image& src2,
                        const gpu_image& src3, const gpu_image& src4 ) {
    gpu_image dst(src0.size(), FMT_FLOAT4);
    launch_config cfg(dst);
    imp_smooth_5<<<cfg.blocks(), cfg.threads()>>>(src0, src1, src2, src3, src4, dst);
    OZ_CUDA_ERROR_CHECK();
    return dst;
}


__global__ void imp_smooth_5half_4( const gpu_plm2<float4> src0, const gpu_plm2<float4> src1,
                                  const gpu_plm2<float4> src2, gpu_plm2<float4> dst) {
    int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float3 c0 = make_float3(src0(ix, iy));
    float3 c1 = make_float3(src1(ix, iy));
    float3 c2 = make_float3(src2(ix, iy));

    dst.write(ix, iy, make_float4((c0 + 2*c1 + 4*c2) / 7, 0));
}


__global__ void imp_smooth_5half_1( const gpu_plm2<float> src0, const gpu_plm2<float> src1, const gpu_plm2<float> src2,
                              gpu_plm2<float> dst) {
    int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float c0 = src0(ix, iy);
    float c1 = src1(ix, iy);
    float c2 = src2(ix, iy);

    dst.write(ix, iy, (c0 + 2*c1 + 4*c2) / 7);
}


gpu_image gpu_smooth_5half( const gpu_image& src0, const gpu_image& src1, const gpu_image& src2 ) {
    switch (src0.format()) {
    case FMT_FLOAT4:
    {
        gpu_image dst(src0.size(), FMT_FLOAT4);
        launch_config cfg(dst);
        imp_smooth_5half_4<<<cfg.blocks(), cfg.threads()>>>(src0, src1, src2, dst);
        OZ_CUDA_ERROR_CHECK();
        return dst;
    }
    case FMT_FLOAT: {
        gpu_image dst(src0.size(), FMT_FLOAT);
        launch_config cfg(dst);
        imp_smooth_5half_1<<<cfg.blocks(), cfg.threads()>>>(src0, src1, src2, dst);
        OZ_CUDA_ERROR_CHECK();
        return dst;
    }
    default:
        OZ_INVALID_FORMAT();
    }
}


__global__ void imp_colorize_sign( const gpu_plm2<float> src, gpu_plm2<float4> dst) {
    int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float s = src(ix, iy);
    dst.write(ix, iy, (s < 0)? make_float4(0,1,0,1) : (s > 0)? make_float4(1,0,0,1) : make_float4(0,0,0,1));
}


gpu_image gpu_colorize_sign( const gpu_image& src) {
    gpu_image dst(src.size(), FMT_FLOAT4);
    launch_config cfg(dst);
    imp_colorize_sign<<<cfg.blocks(), cfg.threads()>>>(src, dst);
    OZ_CUDA_ERROR_CHECK();
    return dst;
}


__global__ void imp_warp_by_flow( const gpu_plm2<float> flowU, const gpu_plm2<float> flowV, gpu_plm2<float4> dst) {
    int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float u = flowU(ix,iy);
    float v = flowV(ix,iy);
    float4 c = tex2D(texSRC, ix + 0.5f + u, iy + 0.5 + v);
    dst.write(ix, iy, c);
}


gpu_image gpu_warp_by_flow( const gpu_image& src,
                            const gpu_image& flowU, const gpu_image& flowV ) {
    gpu_image dst(src.size(), FMT_FLOAT4);
    gpu_binder<float4> src_(texSRC, src);
    launch_config cfg(dst);
    imp_warp_by_flow<<<cfg.blocks(), cfg.threads()>>>(flowU, flowV, dst);
    OZ_CUDA_ERROR_CHECK();
    return dst;
}


__global__ void imp_smooth_10half( const gpu_plm2<float4> src0, const gpu_plm2<float4> src1, const gpu_plm2<float4> src2,
                                   const gpu_plm2<float4> src3, const gpu_plm2<float4> src4, gpu_plm2<float4> dst) {
    int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if(ix >= dst.w || iy >= dst.h)
        return;

    float3 c0 = make_float3(src0(ix, iy));
    float3 c1 = make_float3(src1(ix, iy));
    float3 c2 = make_float3(src2(ix, iy));
    float3 c3 = make_float3(src2(ix, iy));
    float3 c4 = make_float3(src2(ix, iy));

    dst.write(ix, iy, make_float4((70*c0 + 54*c1 + 28*c2 + 8*c3 + c4) / 161, 0));
}


gpu_image gpu_smooth_10half( const gpu_image& src0, const gpu_image& src1, const gpu_image& src2,
                             const gpu_image& src3, const gpu_image& src4 ) {
    gpu_image dst(src0.size(), FMT_FLOAT4);
    launch_config cfg(dst);
    imp_smooth_10half<<<cfg.blocks(), cfg.threads()>>>(src0, src1, src2, src3, src4, dst);
    OZ_CUDA_ERROR_CHECK();
    return dst;
}


__global__ void imp_nagmat_sst(gpu_plm2<float4> dst, size_t krnl_size, float radius, float q) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    const float PI = 3.14159265358979323846f;
    const int N = 8;

    float4 m[N];
    #pragma unroll
    for (int k = 0; k < N; ++k) {
        m[k] = make_float4(0);
    }

    float piN = 2 * PI / float(N);
    float4 X = make_float4(cosf(piN), sinf(piN), -sinf(piN), cosf(piN));

    for (int j = -radius; j <= radius; ++j) {
        for (int i = -radius; i <= radius; ++i) {
            float2 v = make_float2( 0.5f * float(i) / float(radius),
                                    0.5f * float(j) / float(radius));

            if (dot(v,v) <= 0.25f) {
                float3 c = make_float3(tex2D(texSRC, ix + i, iy + j));

                for (int k = 0; k < N; ++k) {
                    float wx = tex2D(texKRNL, krnl_size * (v.x + 0.5f), krnl_size * (v.y + 0.5f));

                    m[k] += make_float4(c * wx, wx);

                    v = make_float2(X.x * v.x + X.z * v.y,
                                    X.y * v.x + X.w * v.y);
                }
            }
        }
    }

    float4 o = make_float4(0);
    #pragma unroll
    for (int k = 0; k < N; ++k) {
        float3 g = make_float3(m[k].x / m[k].w, m[k].y / m[k].w, m[k].z / m[k].w);

        float lambda2 = fmaxf(0, 0.5f * (g.y + g.x -
            sqrtf(fmaxf(0, g.y*g.y - 2*g.x*g.y + g.x*g.x + 4*g.z*g.z))));

        float w = 1.0f / (0.0001f + __powf(lambda2, q));
        o += make_float4(g * w,  w);
    }
    if (o.w == 0) {
        dst.write(ix, iy, make_float4(make_float3(tex2D(texSRC, ix, iy)), 1));
    } else {
        dst.write(ix, iy, make_float4( o.x / o.w, o.y / o.w, o.z / o.w, 1));
    }
}


gpu_image gpu_nagmat_sst( const gpu_image& st, const gpu_image& krnl, float radius, float q ) {
    gpu_image dst(st.size(), FMT_FLOAT4);
    gpu_binder<float4> (texSRC, st);
    gpu_binder<float> (texKRNL, krnl);
    launch_config cfg(dst);
    imp_nagmat_sst<<<cfg.blocks(), cfg.threads()>>>(dst, krnl.w(), radius, q);
    OZ_CUDA_ERROR_CHECK();
    return dst;
}


__global__ void imp_gradient_bf( gpu_plm2<float4> dst, gpu_plm2<float4> st,
                                 float sigma_d, float sigma_r, bool adaptive ) {
    const int ix = __mul24(blockDim.x, blockIdx.x) + threadIdx.x;
    const int iy = __mul24(blockDim.y, blockIdx.y) + threadIdx.y;
    if (ix >= dst.w || iy >= dst.h)
        return;

    float3 s = make_float3(st(ix, iy));
    float2 t = st2tangent(s);
    t = make_float2(t.y, -t.x);
    float A = st2A(s);
    float2 tabs = fabs(t);
    float ds = 1.0f / ((tabs.x > tabs.y)? tabs.x : tabs.y);

    float Asigma_d = adaptive? (1 + sqrtf(A)) * sigma_d : sigma_d;
    float twoSigmaD2 = 2.0f * Asigma_d * Asigma_d;
    float twoSigmaR2 = 2.0f * sigma_r * sigma_r;
    int halfWidth = int(ceilf( 3.0f * sigma_d ));

    float4 c0 = tex2D(texSRC, 0.5f + ix, 0.5f + iy);
    float4 sum = c0;

    float norm = 1;
    for (float d = ds; d <= halfWidth; d += ds) {
        float2 dt = d * t;
        float4 c1 = tex2D(texSRC, 0.5f + ix + dt.x, 0.5f + iy + dt.y);
        float4 c2 = tex2D(texSRC, 0.5f + ix - dt.x, 0.5f + iy - dt.y);

        float4 e1 = c1 - c0;
        float4 e2 = c2 - c0;

        float kd = __expf( -dot(d,d) / twoSigmaD2 );
        float kr1 = __expf( -dot(e1,e1) / twoSigmaR2 );
        float kr2 = __expf( -dot(e2,e2) / twoSigmaR2 );

        sum += kd * kr1 * c1;
        sum += kd * kr2 * c2;
        norm += kd * kr1 + kd * kr2;
    }
    sum /= norm;

    dst.write(ix, iy, sum);
}


gpu_image gradient_bf( const gpu_image& src, const gpu_image& st,
                       float sigma_d, float sigma_r, bool adaptive )
{
    if (sigma_d <= 0) return src;
    gpu_image dst(src.size(), FMT_FLOAT4);
    gpu_binder<float4> src_(texSRC, src, cudaFilterModeLinear);
    launch_config cfg(dst);
    imp_gradient_bf<<<cfg.blocks(), cfg.threads()>>>(dst, st, sigma_d, sigma_r, adaptive);
    OZ_CUDA_ERROR_CHECK();
    return dst;
}
