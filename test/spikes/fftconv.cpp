// by Jan Eric Kyprianidis <www.kyprianidis.com>
#include "fftconv.h"
#include <oz/gkf_kernel.h>
#include <oz/gauss.h>
#include <oz/minmax.h>
#include <oz/colormap.h>
#include <oz/color.h>
#include <oz/fft.h>
#include <oz/norm.h>
#include <oz/conv.h>
#include <cufft.h>
using namespace oz;


FFTConvTest::FFTConvTest() {
    new ParamInt   (this, "N", 8, 2, 8, 1, &N_);
    new ParamInt   (this, "k", 0, 0, 7, 1, &k_);
    new ParamDouble(this, "sigma_r", 3, 0, 100, 0.5, &sigma_r_);
}


void FFTConvTest::process() {
    gpu_image src = rgb2gray(gpuInput0());

    unsigned hw = ceilf(3 * sigma_r_);
    unsigned ksize = 2 * hw + 1;
    cpu_image ck(ksize, ksize, FMT_FLOAT);
    for (unsigned j = 0; j < ksize; ++j) {
        for (unsigned i = 0; i < ksize; ++i) {
            int x = i - hw;
            int y = j - hw;
            float g = expf(-0.5f * (x*x + y*y) / (sigma_r_ * sigma_r_));
            ck.at<float>(i,j) = g;
        }
    }

    gpu_image krnl = ck;
    krnl = krnl / sum(krnl);
    publish("krnl3", colormap_jet(normalize(krnl)));

    {   
        unsigned w = src.w() + ksize-1;
        unsigned h = src.h() + ksize-1;
        cufftHandle planf,plani;
        cufftReal *sdata, *kdata;
        cufftComplex *sspec, *kspec;

        cudaMalloc((void**)&kdata, 4 * w * h);
        cudaMalloc((void**)&sspec, 8 * (w/2+1) * h);
        cudaMalloc((void**)&kspec, 8 * (w/2+1) * h);
        
        cufftResult_t result;
        result = cufftPlan2d(&planf, h, w, CUFFT_R2C);
        assert(result == CUFFT_SUCCESS);
        result = cufftPlan2d(&plani, h, w, CUFFT_C2R);
        assert(result == CUFFT_SUCCESS);

        cudaMalloc(&kdata, 4 * w * h);
        fft_pad_shift(kdata, w, h, krnl);
        cufftExecR2C(planf, kdata, kspec);
        {
            gpu_image kd(w, h, FMT_FLOAT);
            cudaMemcpy2D(kd.ptr(), kd.pitch(), kdata, 4*w, 4*w, h, cudaMemcpyDeviceToDevice);
            publish("kdata", normalize(kd));
        }
        
        cudaMalloc(&sdata, 4 * w * h);
        fft_pad_wrap(sdata, w, h, src);
        {
            gpu_image sd(w, h, FMT_FLOAT);
            cudaMemcpy2D(sd.ptr(), sd.pitch(), sdata, 4*w, 4*w, h, cudaMemcpyDeviceToDevice);
            publish("sdata", sd);
        }

        cufftExecR2C(planf, sdata, sspec);
        fft_complex_norm_mul(sspec, sspec, kspec, w * h, (w/2+1) * h);
        cufftExecC2R(plani, sspec, sdata);

        gpu_image dst(src.w(), src.h(), FMT_FLOAT);
        cudaMemcpy2D(dst.ptr(), dst.pitch(), sdata, 4*w, 4*src.w(), src.h(), cudaMemcpyDeviceToDevice);

        cufftDestroy(planf);
        cufftDestroy(plani);
        cudaFree(sdata); 
        cudaFree(kdata); 
        cudaFree(kspec);
        cudaFree(sspec);

        publish("dst", dst);

        {
            gpu_image dst2 = gauss_filter_xy(src, sigma_r_, 3);
            publish("dst2", dst2);
            publish("diff", colormap_diff(dst, dst2, 1e-4f));
        }
    }


    {
        gpu_image C = conv(krnl, src);
        publish("dstC", C);
    }
}


