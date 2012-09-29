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
#include <oz/hist.h>
#include <oz/color.h>
#include <npp.h>
#include <cfloat>


#define NPP_CHECK_CUDA(S) do {cudaError_t eCudaResult; \
                              eCudaResult = S; \
                              assert(eCudaResult == cudaSuccess);} while (false)

#define NPP_CHECK_NPP(S) do {NppStatus eStatusNPP; \
                             eStatusNPP = S; \
                             assert(eStatusNPP == NPP_SUCCESS);} while (false)


std::vector<int> oz::hist( const gpu_image& src ) {
    if (src.format() == FMT_FLOAT) return hist(src, 256, 0, 1);
    if (src.format() != FMT_UCHAR) OZ_INVALID_FORMAT();

    const int binCount = 256;
    const int levelCount = binCount + 1;

    int bufferSize;
    nppiHistogramEvenGetBufferSize_8u_C1R(src.size(), levelCount, &bufferSize);
    Npp8u * pDeviceBuffer;
    NPP_CHECK_CUDA(cudaMalloc((void **)&pDeviceBuffer, bufferSize));

    Npp32s * histDevice = 0;
    NPP_CHECK_CUDA(cudaMalloc((void **)&histDevice,   binCount   * sizeof(Npp32s)));

    NPP_CHECK_NPP(nppiHistogramEven_8u_C1R(src.ptr<uchar>(), src.pitch(), src.size(),
                                           histDevice, levelCount, 0, binCount,
                                           pDeviceBuffer));

    std::vector<int> H(binCount);
    NPP_CHECK_CUDA(cudaMemcpy(&H[0], histDevice, binCount * sizeof(Npp32s), cudaMemcpyDeviceToHost));
    cudaFree(histDevice);
    cudaFree(pDeviceBuffer);
    return H;
}


std::vector<int> oz::hist( const gpu_image& src, int nbins, float pmin, float pmax) {
    if (src.format() != FMT_FLOAT) OZ_INVALID_FORMAT();
    pmin -= FLT_EPSILON;
    pmax += FLT_EPSILON;

    //NppiSize sizeROI = { src.w(), src.h() };
    int bufferSize;
    nppiHistogramRangeGetBufferSize_32f_C1R(src.size(), nbins + 1, &bufferSize);
    Npp8u * pDeviceBuffer;
    NPP_CHECK_CUDA(cudaMalloc((void**)&pDeviceBuffer, bufferSize));

    Npp32f *levelsDevice = 0;
    NPP_CHECK_CUDA(cudaMalloc((void**)&levelsDevice, (nbins + 1)* sizeof(Npp32f)));
    Npp32f *levelsHost = new Npp32f[nbins + 1];
    for (int i = 0; i < nbins + 1; ++i) {
        levelsHost[i] = 1.0f * ((nbins - i) * pmin +  i * pmax) / nbins;
    }
    NPP_CHECK_CUDA(cudaMemcpy(levelsDevice, levelsHost, (nbins + 1) * sizeof(Npp32f), cudaMemcpyHostToDevice));

    Npp32s * histDevice = 0;
    NPP_CHECK_CUDA(cudaMalloc((void **)&histDevice, nbins * sizeof(Npp32s)));

    NPP_CHECK_NPP(nppiHistogramRange_32f_C1R(src.ptr<float>(), src.pitch(), src.size(),
                                             histDevice, levelsDevice, nbins + 1,
                                             pDeviceBuffer));

    std::vector<int> H(nbins);
    NPP_CHECK_CUDA(cudaMemcpy(&H[0], histDevice, nbins * sizeof(Npp32s), cudaMemcpyDeviceToHost));
    cudaFree(histDevice);
    cudaFree(levelsDevice);
    cudaFree(pDeviceBuffer);
    return H;
}


std::vector<int> oz::hist_join( const std::vector<int>& H0, const std::vector<int>& H1 ) {
    if (H0.size() && !H1.size()) return H0;
    if (H1.size() && !H0.size()) return H1;
    if (H0.size() != H1.size()) OZ_INVALID_SIZE();
    std::vector<int> H;
    H.reserve(H0.size());
    for (int i = 0; i < H0.size(); ++i) {
        H.push_back(H0[i] + H1[i]);
    }
    return H;
}


void oz::hist_insert( std::vector<int>& H, float value, float pmin, float pmax) {
    int index = (int)(H.size() * (value - pmin) / (pmax - pmin));
    if (index < 0) index = 0;
    if (index > H.size()-1) index = (int)H.size()-1;
    ++H[index];
}


std::vector<float> oz::hist_to_pdf( const std::vector<int>& H, float a, float b ) {
    std::vector<float> pdf;
    size_t N = H.size();
    if (N < 2) {
        pdf.push_back(1);
    } else {
        float dx = (b - a) / N;

        float A = 0;
        for (size_t i = 0; i < N; ++i) {
            A += H[i] * dx;
        }

        if (A > 0) {
            pdf.reserve(N);
            for (size_t i = 0; i < N; ++i) {
                pdf.push_back(H[i] / A);
            }
        } else {
            pdf = std::vector<float>(N, 0.0f);
        }
    }
    return pdf;
}


std::vector<float> oz::pdf_to_cdf( const std::vector<float>& pdf, float a, float b ) {
    std::vector<float> cdf;
    cdf.reserve(pdf.size());
    float dx = (b - a) / pdf.size();
    float A = 0;
    for (size_t i = 0; i < pdf.size(); ++i) {
        float dA = pdf[i] * dx;
        A += dA;
        cdf.push_back(A);
    }
    return cdf;
}


float oz::pdf_sgnf( const std::vector<float>& pdf, float a, float b, float s ) {
    float dx = (b - a) / pdf.size();
    float x = a;
    float A = 0;
    for (size_t i = 0; i < pdf.size(); ++i) {
        float dA = pdf[i] * dx;
        if (A + dA > s) {
            x += (s - A) * dx / dA;
            return x;
        }
        A += dA;
        x += dx;
    }
    return b;
}


oz::gpu_image oz::hist_eq( const gpu_image& src ) {
    if (src.format() == FMT_FLOAT) {
        gpu_image usrc = src.convert(FMT_UCHAR);
        gpu_image ueq = hist_eq(usrc);
        gpu_image feq = ueq.convert(FMT_FLOAT);
        return ueq;
    }
    if (src.format() != FMT_UCHAR) OZ_INVALID_FORMAT();

    const int binCount = 256;
    const int levelCount = binCount + 1;

    int bufferSize;
    nppiHistogramEvenGetBufferSize_8u_C1R(src.size(), levelCount, &bufferSize);

    Npp32s * histDevice = 0;
    Npp32s * levelsDevice = 0;

    NPP_CHECK_CUDA(cudaMalloc((void **)&histDevice,   binCount   * sizeof(Npp32s)));
    NPP_CHECK_CUDA(cudaMalloc((void **)&levelsDevice, levelCount * sizeof(Npp32s)));

    Npp8u * pDeviceBuffer;
    NPP_CHECK_CUDA(cudaMalloc((void **)&pDeviceBuffer, bufferSize));

    Npp32s levelsHost[levelCount];
    NPP_CHECK_NPP(nppiEvenLevelsHost_32s(levelsHost, levelCount, 0, binCount));

    NPP_CHECK_NPP(nppiHistogramEven_8u_C1R(src.ptr<uchar>(), src.pitch(), src.size(),
                                           histDevice, levelCount, 0, binCount,
                                           pDeviceBuffer));

    Npp32s histHost[binCount];
    NPP_CHECK_CUDA(cudaMemcpy(histHost, histDevice, binCount * sizeof(Npp32s), cudaMemcpyDeviceToHost));
    cudaFree(histDevice);
    cudaFree(levelsDevice);
    cudaFree(pDeviceBuffer);

    Npp32s  lutHost[binCount + 1];
    {
        Npp32s * pHostHistogram = histHost;
        Npp32s totalSum = 0;
        for (; pHostHistogram < histHost + binCount; ++pHostHistogram)
            totalSum += *pHostHistogram;

        assert(totalSum == src.w() * src.h());

        if (totalSum == 0)
            totalSum = 1;
        float multiplier = 1.0f / float(totalSum) * 0xFF;

        Npp32s runningSum = 0;
        Npp32s * pLookupTable = lutHost;
        for (pHostHistogram = histHost; pHostHistogram < histHost + binCount; ++pHostHistogram) {
            *pLookupTable = (Npp32s)(runningSum * multiplier + 0.5f);
            pLookupTable++;
            runningSum += *pHostHistogram;
        }

        lutHost[binCount] = 0xFF; // last element is always 1
    }

    gpu_image dst(src.size(), FMT_UCHAR);
    NppStatus status  = nppiLUT_Linear_8u_C1R(src.ptr<uchar>(), src.pitch(),
                                        dst.ptr<uchar>(), dst.pitch(), dst.size(),
                                        lutHost, // value and level arrays are in host memory
                                        levelsHost,
                                        binCount+1 );

    return dst;
}


oz::gpu_image oz::hist_auto_levels( const gpu_image& src, float threshold ) {
    std::vector<int> H;
    switch (src.format()) {
        case FMT_FLOAT:
            H = hist(src);
            break;
        case FMT_FLOAT3:
            H = hist(rgb2gray(src));
            break;
        default:
            OZ_INVALID_FORMAT();
    }

    int N = (int)(src.w() * src.h() * threshold / 100);
    int lmin = 0;
    {
        int sum = 0;
        while ((lmin < 256) && (sum < N)) sum += H[lmin++];
    }
    int lmax = 255;
    {
        int sum = 0;
        while ((lmax >= 0) && (sum < N)) sum += H[lmax--];
    }

    float pmin = lmin / 255.0f;
    float pmax = lmax / 255.0f;
    if (pmin > 0.45f) pmin = 0.55f;
    if (pmax < 0.55f) pmax = 0.55f;
    float a = 1.0f / (pmax - pmin);
    float b = -pmin / (pmax - pmin);
    return adjust(src, a, b);
}


std::vector<float> oz::pdf( const gpu_image& src, int nbins, float pmin, float pmax ) {
    std::vector<int> H = hist(src, nbins, pmin, pmax);
    return hist_to_pdf(H, pmin, pmax);
}
