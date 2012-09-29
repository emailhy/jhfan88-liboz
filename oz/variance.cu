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
#include <oz/variance.h>
#include <oz/gpu_plm2.h>


template <unsigned nThreads, unsigned nBlocks>
__global__ void impl_mean( float *dst, const oz::gpu_plm2<float> src ) {
    __shared__ float sdata[nThreads];

    unsigned int t = threadIdx.x;
    float my = 0;

    unsigned o = blockIdx.x * src.stride;
    while (o < src.h * src.stride) {
        unsigned int i = threadIdx.x;
        while (i < src.w) {
            my += src.ptr[o+i];
            i += nThreads;
        }
        o += nBlocks * src.stride;
    }

    sdata[t] = my;
    __syncthreads();

    if (nThreads >= 512) { if (t < 256) { sdata[t] = my = my + sdata[t + 256]; } __syncthreads(); }
    if (nThreads >= 256) { if (t < 128) { sdata[t] = my = my + sdata[t + 128]; } __syncthreads(); }
    if (nThreads >= 128) { if (t <  64) { sdata[t] = my = my + sdata[t +  64]; } __syncthreads(); }

    if (t < 32) {
        volatile float* smem = sdata;
        if (nThreads >=  64) { smem[t] = my = my + smem[t + 32]; }
        if (nThreads >=  32) { smem[t] = my = my + smem[t + 16]; }
        if (nThreads >=  16) { smem[t] = my = my + smem[t +  8]; }
        if (nThreads >=   8) { smem[t] = my = my + smem[t +  4]; }
        if (nThreads >=   4) { smem[t] = my = my + smem[t +  2]; }
        if (nThreads >=   2) { smem[t] = my = my + smem[t +  1]; }
    }

    if (t == 0) {
        dst[blockIdx.x] = sdata[0];
    }
}


float oz::mean( const gpu_image& src ) {
    if (src.format() != FMT_FLOAT) OZ_INVALID_FORMAT();
    const unsigned nBlocks = 64;
    const unsigned nThreads = 128;

    static float *dst_gpu = 0;
    static float *dst_cpu = 0;
    if (!dst_cpu) {
        cudaMalloc(&dst_gpu, sizeof(float)*nBlocks);
        cudaMallocHost(&dst_cpu, sizeof(float)*nBlocks, cudaHostAllocPortable);
    }

    dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
    impl_mean<nThreads, nBlocks><<< dimGrid, dimBlock >>>(dst_gpu, src);
    cudaMemcpy(dst_cpu, dst_gpu, sizeof(float)*nBlocks, cudaMemcpyDeviceToHost);

    int N = src.w() * src.h();
    float mean = 0;
    for (int i = 0; i < nBlocks; ++i) mean += dst_cpu[i];
    return mean / N;
}


template <unsigned nThreads, unsigned nBlocks>
__global__ void impl_variance( float *dst, const oz::gpu_plm2<float> src ) {
    __shared__ float sdata[2 * nThreads];

    unsigned int t1 = threadIdx.x;
    unsigned int t2 = threadIdx.x + nThreads;
    float myM1 = 0;
    float myM2 = 0;

    unsigned o = blockIdx.x * src.stride;
    while (o < src.h * src.stride) {
        unsigned int i = threadIdx.x;
        while (i < src.w) {
            volatile float v = src.ptr[o+i];
            myM1 += v;
            myM2 += v * v;
            i += nThreads;
        }
        o += nBlocks * src.stride;
    }

    sdata[t1] = myM1;
    sdata[t2] = myM2;
    __syncthreads();

    if (nThreads >= 512) { if (t1 < 256) { sdata[t1] = myM1 = myM1 + sdata[t1 + 256];
                                           sdata[t2] = myM2 = myM2 + sdata[t2 + 256]; } __syncthreads(); }
    if (nThreads >= 256) { if (t1 < 128) { sdata[t1] = myM1 = myM1 + sdata[t1 + 128];
                                           sdata[t2] = myM2 = myM2 + sdata[t2 + 128]; } __syncthreads(); }
    if (nThreads >= 128) { if (t1 <  64) { sdata[t1] = myM1 = myM1 + sdata[t1 +  64];
                                           sdata[t2] = myM2 = myM2 + sdata[t2 +  64]; } __syncthreads(); }

    if (t1 < 32) {
        volatile float* smem = sdata;
        if (nThreads >=  64) { smem[t1] = myM1 = myM1 + smem[t1 + 32];
                               smem[t2] = myM2 = myM2 + smem[t2 + 32]; }
        if (nThreads >=  32) { smem[t1] = myM1 = myM1 + smem[t1 + 16];
                               smem[t2] = myM2 = myM2 + smem[t2 + 16]; }
        if (nThreads >=  16) { smem[t1] = myM1 = myM1 + smem[t1 +  8];
                               smem[t2] = myM2 = myM2 + smem[t2 +  8]; }
        if (nThreads >=   8) { smem[t1] = myM1 = myM1 + smem[t1 +  4];
                               smem[t2] = myM2 = myM2 + smem[t2 +  4]; }
        if (nThreads >=   4) { smem[t1] = myM1 = myM1 + smem[t1 +  2];
                               smem[t2] = myM2 = myM2 + smem[t2 +  2]; }
        if (nThreads >=   2) { smem[t1] = myM1 = myM1 + smem[t1 +  1];
                               smem[t2] = myM2 = myM2 + smem[t2 +  1]; }
    }

    if (t1 == 0) {
        dst[blockIdx.x] = sdata[0];
        dst[blockIdx.x + nBlocks] = sdata[0 + nThreads];
    }
}


void oz::variance(const gpu_image& src, float *mean, float *variance ) {
    if (src.format() != FMT_FLOAT) OZ_INVALID_FORMAT();
    const unsigned nBlocks = 64;
    const unsigned nThreads = 128;

    static float *dst_gpu = 0;
    static float *dst_cpu = 0;
    if (!dst_cpu) {
        cudaMalloc(&dst_gpu, 2 * sizeof(float)*nBlocks);
        cudaMallocHost(&dst_cpu, 2 * sizeof(float)*nBlocks, cudaHostAllocPortable);
    }

    dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
    impl_variance<nThreads, nBlocks><<< dimGrid, dimBlock >>>(dst_gpu, src);
    cudaMemcpy(dst_cpu, dst_gpu, 2*sizeof(float)*nBlocks, cudaMemcpyDeviceToHost);

    int N = src.w() * src.h();
    float m1 = 0;
    if (mean || variance) {
        for (int i = 0; i < nBlocks; ++i) m1 += dst_cpu[i];
        m1 /= N;
        if (mean) *mean = m1;
    }
    if (variance) {
        float m2 = 0;
        for (int i = 0; i < nBlocks; ++i) m2 += dst_cpu[nBlocks+i];
        *variance = fabs((float)m2 / N - m1 * m1);
    }
}


template <unsigned nThreads, unsigned nBlocks>
__global__ void impl_covariance( float *dst, const oz::gpu_plm2<float> src0, const oz::gpu_plm2<float> src1 ) {
    __shared__ float sdata[3 * nThreads];

    unsigned int t1 = threadIdx.x;
    unsigned int t2 = threadIdx.x + nThreads;
    unsigned int t3 = threadIdx.x + 2 * nThreads;
    float E0 = 0;
    float E1 = 0;
    float E01 = 0;

    unsigned o0 = blockIdx.x * src0.stride;
    unsigned o1 = blockIdx.x * src1.stride;
    while (o0 < src0.h * src0.stride) {
        unsigned int i = threadIdx.x;
        while (i < src0.w) {
            volatile float v0 = src0.ptr[o0 + i];
            volatile float v1 = src1.ptr[o1 + i];
            E0 += v0;
			E1 += v1;
            E01 += v0 * v1;
            i += nThreads;
        }
        o0 += nBlocks * src0.stride;
        o1 += nBlocks * src1.stride;
    }

    sdata[t1] = E0;
    sdata[t2] = E1;
    sdata[t3] = E01;
    __syncthreads();

    if (nThreads >= 512) { if (t1 < 256) { sdata[t1] = E0  = E0  + sdata[t1 + 256];
                                           sdata[t2] = E1  = E1  + sdata[t2 + 256];
                                           sdata[t3] = E01 = E01 + sdata[t3 + 256]; } __syncthreads(); }
    if (nThreads >= 256) { if (t1 < 128) { sdata[t1] = E0  = E0  + sdata[t1 + 128];
                                           sdata[t2] = E1  = E1  + sdata[t2 + 128];
                                           sdata[t3] = E01 = E01 + sdata[t3 + 128]; } __syncthreads(); }
    if (nThreads >= 128) { if (t1 <  64) { sdata[t1] = E0 =  E0  + sdata[t1 +  64];
                                           sdata[t2] = E1 =  E1  + sdata[t2 +  64];
                                           sdata[t3] = E01 = E01 + sdata[t3 +  64]; } __syncthreads(); }

    if (t1 < 32) {
        volatile float* smem = sdata;
        if (nThreads >=  64) { smem[t1] = E0  = E0  + smem[t1 + 32];
                               smem[t2] = E1  = E1  + smem[t2 + 32];
                               smem[t3] = E01 = E01 + smem[t3 + 32]; }
        if (nThreads >=  32) { smem[t1] = E0 =  E0  + smem[t1 + 16];
                               smem[t2] = E1 =  E1  + smem[t2 + 16];
                               smem[t3] = E01 = E01 + smem[t3 + 16]; }
        if (nThreads >=  16) { smem[t1] = E0 =  E0  + smem[t1 +  8];
                               smem[t2] = E1 =  E1  + smem[t2 +  8];
                               smem[t3] = E01 = E01 + smem[t3 +  8]; }
        if (nThreads >=   8) { smem[t1] = E0  = E0  + smem[t1 +  4];
                               smem[t2] = E1  = E1  + smem[t2 +  4];
                               smem[t3] = E01 = E01 + smem[t3 +  4]; }
        if (nThreads >=   4) { smem[t1] = E0  = E0  + smem[t1 +  2];
                               smem[t2] = E1  = E1  + smem[t2 +  2];
                               smem[t3] = E01 = E01 + smem[t2 +  2]; }
        if (nThreads >=   2) { smem[t1] = E0  = E0  + smem[t1 +  1];
                               smem[t2] = E1  = E1  + smem[t2 +  1];
                               smem[t3] = E01 = E01 + smem[t3 +  1]; }
    }

    if (t1 == 0) {
        dst[blockIdx.x] = sdata[0];
        dst[blockIdx.x + nBlocks] = sdata[0 + nThreads];
        dst[blockIdx.x + 2*nBlocks] = sdata[0 + 2*nThreads];
    }
}


float oz::covariance( const gpu_image& src0, const gpu_image& src1 ) {
    if ((src0.format() != FMT_FLOAT) || (src1.format() != FMT_FLOAT)) OZ_INVALID_FORMAT();
    if (src0.size() != src1.size()) OZ_INVALID_SIZE();

    const unsigned nBlocks = 64;
    const unsigned nThreads = 128;

    static float *dst_gpu = 0;
    static float *dst_cpu = 0;
    if (!dst_cpu) {
        cudaMalloc(&dst_gpu, 3 * sizeof(float)*nBlocks);
        cudaMallocHost(&dst_cpu, 3 * sizeof(float)*nBlocks, cudaHostAllocPortable);
    }

    dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
    impl_covariance<nThreads, nBlocks><<< dimGrid, dimBlock >>>(dst_gpu, src0, src1);
    cudaMemcpy(dst_cpu, dst_gpu, 3*sizeof(float)*nBlocks, cudaMemcpyDeviceToHost);

    int N = src0.w() * src0.h();
    float E0 = 0;
    float E1 = 0;
    float E01 = 0;
    for (int i = 0; i < nBlocks; ++i) {
        E0 += dst_cpu[i];
        E1 += dst_cpu[i + nBlocks];
        E01 += dst_cpu[i + 2 * nBlocks];
    }
    return (E01 - E0*E1) / N;
}


float oz::snr( const gpu_image& src, const gpu_image& src_noise ) {
    float Ps, Pn;
    float ms, mn;
    gpu_image n = src - src_noise;
    variance(src, &ms, &Ps);
    variance(n, &mn, &Pn);
    return (Pn > 0)? 10.0f * log10f(Ps / Pn) : CUDART_NORM_HUGE_F;
}
