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
#include <oz/fftgkf.h>
#include <oz/gkf_kernel.h>
#include <oz/gauss.h>
#include <oz/norm.h>
#include <oz/fft.h>
#include <oz/launch_config.h>
#include <oz/gpu_plm2.h>
#include <oz/shuffle.h>
#include <oz/minmax.h>
#include <oz/log.h>
#include <oz/conv.h>
#include <vector>


namespace oz {
    #if 0
    //Align a to nearest higher multiple of b
    inline int iAlignUp(int a, int b){
        return (a % b != 0) ?  (a - a % b + b) : a;
    }

    // from convolutionFFT2D NVIDIA GPU Computing SDK example
    static int snap_transform_size(int dataSize) {
        int hiBit;
        unsigned int lowPOT, hiPOT;

        dataSize = iAlignUp(dataSize, 16);

        for(hiBit = 31; hiBit >= 0; hiBit--)
            if(dataSize & (1U << hiBit)) break;

        lowPOT = 1U << hiBit;
        if(lowPOT == (unsigned int)dataSize)
            return dataSize;

        hiPOT = 1U << (hiBit + 1);
        if(hiPOT <= 1024)
            return hiPOT;
        else
            return iAlignUp(dataSize, 512);
    }
    #endif


    static int snap_transform_size_16(int dataSize) {
        return 16 * ((dataSize + 15) / 16);
    }


    __global__ void imp_fftgkf_compute( gpu_plm2<float3> dst, float *res, unsigned fw, unsigned fh,
                                        float q, float threshold )
    {
        int ix = blockDim.x * blockIdx.x + threadIdx.x;
        int iy = blockDim.y * blockIdx.y + threadIdx.y;
        if (ix >= dst.w || iy >= dst.h) return;
        int idx = ix + iy * fw;

        float3 o = make_float3(0);
        float ow = 0;
        for (int k = 0; k < 8; ++k) {
            float3 m1 = make_float3(
                res[idx + (6*k+0) * fw * fh],
                res[idx + (6*k+1) * fw * fh],
                res[idx + (6*k+2) * fw * fh]
            );
            float3 m2 = make_float3(
                res[idx + (6*k+3) * fw * fh],
                res[idx + (6*k+4) * fw * fh],
                res[idx + (6*k+5) * fw * fh]
            );

            m2 = fabs(m2 - m1*m1);
            float sigma2 = fmaxf(threshold, sqrtf(sum(m2)));
            float wk = powf(sigma2, -q);

            o += wk * m1;
            ow += wk;
        }
        dst.write(ix, iy, o / ow);
    }


    fftgkf_t::fftgkf_t( unsigned w, unsigned h, float sigma_s, float sigma_r ) {
        const float precision = 3;
        int half_width = (int)ceilf(precision * sigma_r);
        unsigned ksize = 2 * half_width + 1;

        fw_ = w + ksize - 1;
        fh_ = h + ksize - 1;
        fw_ = snap_transform_size_16(fw_);
        fh_ = snap_transform_size_16(fh_);

        cufftPlan2d(&planf_, fh_, fw_, CUFFT_R2C);
        cufftPlan2d(&plani_, fh_, fw_, CUFFT_C2R);

        cufftReal *kdata;
        cudaMalloc(&kdata, 4 * fw_ * fh_);
        for (int k = 0; k < 8; ++k) {
            gpu_image krnl = gkf_char_function(k, 8, 0.5f * ksize, ksize);
            krnl = gauss_filter_xy(krnl, sigma_s, precision);
            krnl = gkf_gaussian_mul(krnl, sigma_r, precision);
            krnl = krnl / sum(krnl);
            //log_image(normalize(krnl), "krnl%d", k);

            fft_pad_shift(kdata, fw_, fh_, krnl);
            cudaMalloc(&kernel_[k], 8 * (fw_/2+1) * fh_);
            cufftExecR2C(planf_, kdata, kernel_[k]);
        }
        cudaFree(kdata);

        cudaMalloc(&spec_, 8 * (fw_/2+1) * fh_);
        cudaMalloc(&tspec_, 8 * (fw_/2+1) * fh_);
        cudaMalloc(&data_, 4 * fw_ * fh_);
        cudaMalloc(&res_, 16 * 3 * 4 * fw_ * fh_);
    }


    gpu_image fftgkf_t::operator()( const gpu_image& src, float threshold, float q ) const {
        for (int c = 0; c < 3; ++c) {
            fft_pad_wrap(data_, fw_, fh_, shuffle(src, c));
            cufftExecR2C(planf_, data_, spec_);
            for (int k = 0; k < 8; ++k) {
                fft_complex_norm_mul(tspec_, spec_, kernel_[k], (float)fw_ * fh_, (fw_/2+1) * fh_);
                cufftExecC2R(plani_, tspec_, &res_[(6*k+c) * fw_ * fh_]);
            }
        }

        gpu_image src2 = sqr(src);
        for (int c = 0; c < 3; ++c) {
            fft_pad_wrap(data_, fw_, fh_, shuffle(src2, c));
            cufftExecR2C(planf_, data_, spec_);
            for (int k = 0; k < 8; ++k) {
                fft_complex_norm_mul(tspec_, spec_, kernel_[k], (float)fw_ * fh_, (fw_/2+1) * fh_);
                cufftExecC2R(plani_, tspec_, &res_[(6*k+3+c) * fw_ * fh_]);
            }
        }

        /*
        gpu_image dst(fw_, 16 * 3 * fh_, FMT_FLOAT);
        cudaMemcpy2D(
            dst.ptr(), dst.pitch(),
            res_,
            4*fw_,
            4*fw_,
            16*3* fh_,
            cudaMemcpyDeviceToDevice );*/

        gpu_image dst(src.w(), src.h(), FMT_FLOAT3);
        launch_config cfg(dst);
        imp_fftgkf_compute<<<cfg.blocks(), cfg.threads()>>>(dst, res_, fw_, fh_, threshold, q);

        return dst;
    }


    fftgkf_t::~fftgkf_t() {
        cufftDestroy(planf_);
        cufftDestroy(plani_);

        for (int k = 0; k < 8; ++k) {
            cudaFree(kernel_[k]);
        }

        cudaFree(spec_);
        cudaFree(tspec_);
        cudaFree(data_);
        cudaFree(res_);
    }


    gpu_image gkf_simple( const gpu_image& src, float sigma_s, float sigma_r ) {
        const float precision = 3;
        int half_width = (int)ceilf(precision * sigma_r);
        unsigned ksize = 2 * half_width + 1;

        gpu_image src2 = sqr(src);
        gpu_image r[16];

        for (int k = 0; k < 8; ++k) {
            gpu_image krnl = gkf_char_function(k, 8, 0.5f * ksize, ksize);
            krnl = gauss_filter_xy(krnl, sigma_s, precision);
            krnl = gkf_gaussian_mul(krnl, sigma_r, precision);
            krnl = krnl / sum(krnl);

            r[2*k + 0] = conv(krnl, src);
            r[2*k + 1] = conv(krnl, src2);

            //log_image(r[2*k + 0], "c%d", k);
            //log_image(r[2*k + 1], "s%d", k);
        }

        float *res;
        cudaMalloc(&res, 16 * 3 * 4 * src.w() * src.h());
        for (int k = 0; k < 16; ++k) {
            for (int c = 0; c < 3; ++c) {
                gpu_image I = shuffle(r[k], c);
                assert(I.w() > 0 && I.h() > 0);
                cudaMemcpy2D(
                    &res[ (3*k+c) * src.w() * src.h() ],
                    4 * src.w(),
                    I.ptr(),
                    I.pitch(),
                    4 * I.w(),
                    I.h(),
                    cudaMemcpyDeviceToDevice );
            }
        }

        /*
        gpu_image dst(src.w(), 16 * 3 * src.h(), FMT_FLOAT);
        cudaMemcpy2D(
            dst.ptr(), dst.pitch(),
            res,
            4*src.w(),
            4*src.w(),
            16*3* src.h(),
            cudaMemcpyDeviceToDevice );
            */

        gpu_image dst(src.w(), src.h(), FMT_FLOAT3);
        launch_config cfg(dst);
        imp_fftgkf_compute<<<cfg.blocks(), cfg.threads()>>>(dst, res, src.w(), src.h(), 8, 1e-4f);

        cudaFree(res);
        return dst;
    }

}
