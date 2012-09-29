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
#include <oz/minmax.h>
#include <oz/gpu_plm2.h>
#include <oz/color.h>
#include <cfloat>


template <unsigned nThreads, unsigned nBlocks>
__global__ void impl_minmax( float *dst, const oz::gpu_plm2<float> src ) {
    __shared__ float sdata[2 * nThreads];

    unsigned int tmin = threadIdx.x;
    unsigned int tmax = threadIdx.x + nThreads;
    float myMin = FLT_MAX;
    float myMax = -FLT_MAX;

    unsigned o = blockIdx.x * src.stride;
    while (o < src.h * src.stride) {
        unsigned int i = threadIdx.x;
        while (i < src.w) {
            volatile float v = src.ptr[o+i];
            myMin = fminf(myMin, v);
            myMax = fmaxf(myMax, v);
            i += nThreads;
        }
        o += nBlocks * src.stride;
    }

    sdata[tmin] = myMin;
    sdata[tmax] = myMax;
    __syncthreads();

    if (nThreads >= 512) { if (tmin < 256) { sdata[tmin] = myMin = fminf(myMin, sdata[tmin + 256]);
                                             sdata[tmax] = myMax = fmaxf(myMax, sdata[tmax + 256]); } __syncthreads(); }
    if (nThreads >= 256) { if (tmin < 128) { sdata[tmin] = myMin = fminf(myMin, sdata[tmin + 128]);
                                             sdata[tmax] = myMax = fmaxf(myMax, sdata[tmax + 128]); } __syncthreads(); }
    if (nThreads >= 128) { if (tmin <  64) { sdata[tmin] = myMin = fminf(myMin, sdata[tmin +  64]);
                                             sdata[tmax] = myMax = fmaxf(myMax, sdata[tmax +  64]); } __syncthreads(); }

    if (tmin < 32) {
        volatile float* smem = sdata;
        if (nThreads >=  64) { smem[tmin] = myMin = fminf(myMin, smem[tmin + 32]);
                               smem[tmax] = myMax = fmaxf(myMax, smem[tmax + 32]); }
        if (nThreads >=  32) { smem[tmin] = myMin = fminf(myMin, smem[tmin + 16]);
                               smem[tmax] = myMax = fmaxf(myMax, smem[tmax + 16]); }
        if (nThreads >=  16) { smem[tmin] = myMin = fminf(myMin, smem[tmin +  8]);
                               smem[tmax] = myMax = fmaxf(myMax, smem[tmax +  8]); }
        if (nThreads >=   8) { smem[tmin] = myMin = fminf(myMin, smem[tmin +  4]);
                               smem[tmax] = myMax = fmaxf(myMax, smem[tmax +  4]); }
        if (nThreads >=   4) { smem[tmin] = myMin = fminf(myMin, smem[tmin +  2]);
                               smem[tmax] = myMax = fmaxf(myMax, smem[tmax +  2]); }
        if (nThreads >=   2) { smem[tmin] = myMin = fminf(myMin, smem[tmin +  1]);
                               smem[tmax] = myMax = fmaxf(myMax, smem[tmax +  1]); }
    }

    if (tmin == 0) {
        dst[blockIdx.x] = sdata[0];
        dst[blockIdx.x + nBlocks] = sdata[0 + nThreads];
    }
}


void oz::minmax( const gpu_image& src, float *pmin, float *pmax ) {
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
    impl_minmax<nThreads, nBlocks><<< dimGrid, dimBlock >>>(dst_gpu, gpu_plm2<float>(src));
    cudaMemcpy(dst_cpu, dst_gpu, 2*sizeof(float)*nBlocks, cudaMemcpyDeviceToHost);

    if (pmin) {
        float m = dst_cpu[0];
        for (int i = 1; i < nBlocks; ++i) m = fminf(m, dst_cpu[i]);
        *pmin = m;
    }
    if (pmax) {
        float m = dst_cpu[nBlocks];
        for (int i = 1; i < nBlocks; ++i) m = fmaxf(m, dst_cpu[nBlocks+i]);
        *pmax = m;
    }
}


float oz::min( const gpu_image& src ) {
    float m;
    minmax(src, &m, NULL);
    return m;
}


float oz::max( const gpu_image& src ) {
    float m;
    minmax(src, NULL, &m);
    return m;
}


oz::gpu_image oz::normalize( const gpu_image& src ) {
    switch (src.format()) {
        case FMT_FLOAT:
            {
                float pmin, pmax;
                minmax(src, &pmin, &pmax);
                return adjust(src, 1.0f / (pmax - pmin), -pmin / (pmax - pmin));
            }
        case FMT_FLOAT3:
            {
                float pmin, pmax;
                gpu_image L = rgb2gray(src);
                minmax(L, &pmin, &pmax);
                return adjust(src, 1.0f / (pmax - pmin), -pmin / (pmax - pmin));
            }
        default:
            OZ_INVALID_FORMAT();
    }
}
