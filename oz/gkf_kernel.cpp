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
#include <oz/gkf_kernel.h>
#include <oz/cpu_image.h>
#include <cstring>


static void gauss_filter(float *data, int width, int height, float sigma) {
    float twoSigma2 = 2.0f * sigma * sigma;
    int halfWidth = (int)ceil( 2.0f * sigma );

    float *src_data = new float[width * height];
    memcpy(src_data, data, width * height * sizeof(float));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float sum = 0;
            float w = 0;

            for ( int i = -halfWidth; i <= halfWidth; ++i ) {
                for ( int j = -halfWidth; j <= halfWidth; ++j ) {
                    int xi = x + i;
                    int yj = y + j;
                    if ((xi >= 0) && (xi < width) && (yj >= 0) && (yj < height)) {
                        float r = sqrt((float)(i * i + j * j));
                        float k = exp( -r *r / twoSigma2 );
                        w += k;
                        sum += k * src_data[ xi + yj * width];
                    }
                }
            }

            data[ x + y * width ] = sum / w;
        }
    }

    delete[] src_data;
}


static void make_sector(float *krnl, int k, int N, int size, float sigma_r, float sigma_s) {
    float *p = krnl;
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < size; ++i) {
            float x = i - 0.5f * size + 0.5f;
            float y = j - 0.5f * size + 0.5f;
            float r = sqrtf(x * x + y * y);

            float a = 0.5f * atan2f(y, x) / CUDART_PI_F + k * 1.0f / N;
            if (a > 0.5)
                a -= 1.0;
            if (a < -0.5)
                a += 1.0;

            if ((fabs(a) <= 0.5 / N) && (r < 0.5 * size)) {
                *p = 1;
            } else {
                *p = 0;
            }

            ++p;
        }
    }

    gauss_filter(krnl, size, size, sigma_s);

    p = krnl;
    float mx = 0.0;
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < size; ++i) {
            float x = i - 0.5f * size + 0.5f;
            float y = j - 0.5f * size + 0.5f;
            float r = sqrtf(x * x + y * y);
            *p *= expf(-0.5f * r * r / sigma_r / sigma_r);
            if (*p > mx) mx = *p;
            ++p;
        }
    }

    p = krnl;
    for (int j = 0; j < size; ++j) {
        for (int i = 0; i < size; ++i) {
            *p /= mx;
            ++p;
        }
    }
}


oz::gpu_image oz::gkf_create_kernel1( unsigned krnl_size, float smoothing, float precision, int N ) {
    const float sigma = 0.5f * (krnl_size - 1) / precision;
    float *krnl = new float[krnl_size * krnl_size];
    make_sector(krnl, 0, N, krnl_size, sigma, smoothing * sigma);
    cpu_image dst(krnl, krnl_size*sizeof(float), krnl_size, krnl_size);
    delete[] krnl;
    return dst;
}


oz::gpu_image oz::gkf_create_kernel4(unsigned krnl_size, float smoothing, float precision, int N ) {
    const float sigma = 0.5f * (krnl_size - 1) / precision;
    float *krnl = new float[krnl_size * krnl_size];
    float *krnl4 = new float[4 * krnl_size * krnl_size];
    for (int k = 0; k < 4; ++k) {
        make_sector(krnl, k, N, krnl_size, sigma, smoothing * sigma);
        for (unsigned i = 0; i < krnl_size * krnl_size; ++i) {
            krnl4[4*i+k] = krnl[i];
        }
    }
    cpu_image dst((float4*)krnl4, krnl_size*sizeof(float4), krnl_size, krnl_size, false);
    delete[] krnl4;
    delete[] krnl;
    return dst;
}


oz::gpu_image oz::gkf_create_kernel8x2(unsigned krnl_size, float smoothing, float precision ) {
    const int N = 8;
    const float sigma = 0.5f * (krnl_size - 1) / precision;
    float *krnl = new float[krnl_size * krnl_size];
    float *krnl4 = new float[4 * krnl_size * krnl_size];
    for (int k = 0; k < 4; ++k) {
        make_sector(krnl, 2*k, N, krnl_size, sigma, smoothing * sigma);
        for (unsigned i = 0; i < krnl_size * krnl_size; ++i) {
            krnl4[4*i+k] = krnl[i];
        }
    }
    cpu_image dst((float4*)krnl4, krnl_size*sizeof(float4), krnl_size, krnl_size, false);
    delete[] krnl4;
    delete[] krnl;
    return dst;
}
