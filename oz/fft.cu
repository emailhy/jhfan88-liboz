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
#include <oz/fft.h>
#include <oz/make.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>
#include <cufft.h>


namespace oz {

    template<typename T> struct FFTShift : public generator<T> {
        gpu_sampler<T,0> src_;
        unsigned w_, h_;

        FFTShift( const gpu_image& src, unsigned w, unsigned h ) : src_(src), w_(w), h_(h) {}

        inline __device__ T operator()( int ix, int iy ) const {
            int x = ix - w_/2;
            int y = iy - h_/2;
            if (x < 0) x += w_;
            if (y < 0) y += h_;
            return src_(x, y);
        }
    };

    gpu_image fftshift( const gpu_image& src ) {
        switch (src.format()) {
            case FMT_FLOAT:  return generate(src.size(), FFTShift<float >(src, src.w(), src.h()));
            case FMT_FLOAT2: return generate(src.size(), FFTShift<float2>(src, src.w(), src.h()));
            default:
                OZ_INVALID_FORMAT();
        }
    }


    gpu_image fft2( const gpu_image& src ) {
        if (src.format() != FMT_FLOAT) OZ_INVALID_FORMAT();
        cufftHandle plan;
        cufftComplex *idata;
        cufftComplex *odata;

        gpu_image csrc = make(src, 0);

        cudaMalloc((void**)&idata, 8 * src.N());
        cudaMalloc((void**)&odata, 8 * src.N());
        cudaMemcpy2D(idata, 8*src.w(), csrc.ptr(), csrc.pitch(), 8*src.w(), src.h(), cudaMemcpyDeviceToDevice);

        cufftPlan2d(&plan, src.w(), src.h(), CUFFT_C2C);
        cufftExecC2C(plan, idata, odata, CUFFT_FORWARD);

        gpu_image dst(src.w(), src.h(), FMT_FLOAT2);
        cudaMemcpy2D(dst.ptr(), dst.pitch(), odata, 8*src.w(), 8*src.w(), src.h(), cudaMemcpyDeviceToDevice);

        cufftDestroy(plan);
        cudaFree(idata);
        cudaFree(odata);

        return dst;
    }


    struct PsfPadShift : public generator<float> {
        gpu_plm2<float> psf_;
        unsigned w_, h_;

        PsfPadShift( unsigned w, unsigned h, const gpu_image& psf )
            : w_(w), h_(h), psf_(psf) {}

        inline __device__ float operator()( int ix, int iy ) const {
            int w1 = psf_.w / 2;
            int w2 = psf_.w - w1;
            int h1 = psf_.h / 2;
            int h2 = psf_.h - h1;

            int x;
            if (ix < w2)
                x = ix + w1;
            else if (ix >= w_ - h1)
                x = w1 - w_ + ix;
            else
                return 0;

            int y;
            if (iy < h2)
                y = iy + h1;
            else if (iy >= h_ - h1)
                y = h1 - h_ + iy;
            else
                return 0;

            return psf_(x, y);
        }
    };

    gpu_image psf_padshift( unsigned w, unsigned h, const gpu_image& psf ) {
        return generate(w, h, PsfPadShift(w, h, psf));
    }


    __global__ void imp_fft_complex_mul( cuFloatComplex *c, const cuFloatComplex *a,
                                         const cuFloatComplex *b, float scale, unsigned N )
    {
        const int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= N) return;
        cuComplex ai = a[idx];
        cuComplex bi = b[idx] / scale;
        c[idx] = cuCmulf(ai, bi);
    }

    void fft_complex_norm_mul( cuFloatComplex *c, const cuFloatComplex *a, const cuFloatComplex *b,
                               float scale, unsigned N )
    {
        imp_fft_complex_mul<<<(N+63)/64, 64>>>(c, a, b, scale, N);
    }


    __global__ void imp_fft_pad_wrap( gpu_plm2<float> dst, const gpu_plm2<float> src ) {
        int ix = blockDim.x * blockIdx.x + threadIdx.x;
        int iy = blockDim.y * blockIdx.y + threadIdx.y;
        if (ix >= dst.w || iy >= dst.h) return;

        int sx = ix;
        if (sx >= src.w) {
            if (sx >= (src.w + dst.w)/2)
                sx = 0;
            else
                sx = src.w - 1;
        }

        int sy = iy;
        if (sy >= src.h) {
            if (sy >= (src.h + dst.h)/2)
                sy = 0;
            else
                sy = src.h - 1;
        }

        dst.write(ix, iy, src(sx, sy));
    }

    void fft_pad_wrap( float *dst, unsigned w, unsigned h, const gpu_image& src ) {
        gpu_plm2<float> pdst(dst, w, w, h);
        dim3 threads(8, 8);
        dim3 blocks((w + threads.x-1)/threads.x, (h + threads.y-1)/threads.y);
        imp_fft_pad_wrap<<<blocks,threads>>>(pdst, src);
    }


    __global__ void imp_fft_pad_shift( gpu_plm2<float> dst, const gpu_plm2<float> src ) {
        int ix = blockDim.x * blockIdx.x + threadIdx.x;
        int iy = blockDim.y * blockIdx.y + threadIdx.y;
        if (ix >= dst.w || iy >= dst.h) return;

        int w1 = src.w / 2;
        int w2 = src.w - w1;
        int h1 = src.h / 2;
        int h2 = src.h - h1;

        int sx;
        if (ix < w2)
            sx = ix + w1;
        else if (ix >= dst.w - h1)
            sx = w1 - dst.w + ix;
        else {
            dst.write(ix, iy, 0);
            return;
        }

        int sy;
        if (iy < h2)
            sy = iy + h1;
        else if (iy >= dst.h - h1)
            sy = h1 - dst.h + iy;
        else {
            dst.write(ix, iy, 0);
            return;
        }

        dst.write(ix, iy, src(sx, sy));
    }

    void fft_pad_shift( float *dst, unsigned w, unsigned h, const gpu_image& src ) {
        gpu_plm2<float> pdst(dst, w, w, h);
        dim3 threads(8, 8);
        dim3 blocks((w + threads.x-1)/threads.x, (h + threads.y-1)/threads.y);
        imp_fft_pad_shift<<<blocks,threads>>>(pdst, src);
    }

}
