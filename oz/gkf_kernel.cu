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
#include <oz/generate.h>
#include <oz/transform.h>


namespace oz {

    struct CharFunction : public generator<float> {
        int k_;
        int N_;
        float radius_;
        unsigned size_;

        CharFunction( int k, int N, float radius, unsigned size )
            : k_(k), N_(N), radius_(radius), size_(size) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float x = ix + 0.5f - 0.5f * size_;
            float y = iy + 0.5f - 0.5f * size_;

            float phi = atan2f(y, x);
            float r = sqrtf(x * x + y * y);

            float a = 0.5f * atan2f(y, x) / CUDART_PI_F + k_ * 1.0f / N_;
            if (a > 0.5f)
                a -= 1.0f;
            if (a < -0.5f)
                a += 1.0f;

            if ((fabs(a) <= 0.5f / N_) && (r < radius_)) {
                return 1;
            } else {
                return 0;
            }
        }
    };


    gpu_image gkf_char_function( int k, int N, float radius, unsigned size ) {
        return generate(size, size, CharFunction(k, N, radius, size));
    }


    struct GaussianMul : public generator<float> {
        gpu_plm2<float> src_;
        NppiSize size_;
        float sigma_;
        float precision_;

        GaussianMul( const gpu_image& src, NppiSize size, float sigma, float precision )
            : src_(src), size_(size), sigma_(sigma), precision_(precision) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float twoSigma2 = 2.0f * sigma_ * sigma_;
            int halfWidth = ceilf( precision_ * sigma_ );
            float x = ix + 0.5f - 0.5f * size_.width;
            float y = iy + 0.5f - 0.5f * size_.height;
            float r2 = x*x + y*y;
            float s = src_(ix, iy);
            return s * exp( -r2 / twoSigma2);
        }
    };


    gpu_image gkf_gaussian_mul( const gpu_image& src, float sigma, float precision ) {
        return generate(src.size(), GaussianMul(src, src.size(), sigma, precision));
    }

}