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
#include <oz/norm.h>
#include <oz/gpu_plm2.h>

namespace oz {

    template <unsigned nThreads, unsigned nBlocks>
    __global__ void Sum( float *dst, const gpu_plm2<float> src ) {
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


    float sum( const gpu_image& src ) {
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
        Sum<nThreads, nBlocks><<< dimGrid, dimBlock >>>(dst_gpu, src);
        cudaMemcpy(dst_cpu, dst_gpu, sizeof(float)*nBlocks, cudaMemcpyDeviceToHost);

        float sum = 0;
        for (int i = 0; i < nBlocks; ++i) sum += dst_cpu[i];
        return sum;
    }


    template <unsigned nThreads, unsigned nBlocks>
    __global__ void NormL2( float *dst, const gpu_plm2<float> src ) {
        __shared__ float sdata[nThreads];

        unsigned int t = threadIdx.x;
        float my = 0;

        unsigned o = blockIdx.x * src.stride;
        while (o < src.h * src.stride) {
            unsigned int i = threadIdx.x;
            while (i < src.w) {
                volatile float v = src.ptr[o+i];
                my += v * v;
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


    float norm_l2( const gpu_image& src ) {
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
        NormL2<nThreads, nBlocks><<< dimGrid, dimBlock >>>(dst_gpu, src);
        cudaMemcpy(dst_cpu, dst_gpu, sizeof(float)*nBlocks, cudaMemcpyDeviceToHost);

        float norm2 = 0;
        for (int i = 0; i < nBlocks; ++i) norm2 += dst_cpu[i];
        return sqrtf(norm2);
    }

}
