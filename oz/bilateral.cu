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
#include <oz/bilateral.h>
#include <oz/gpu_sampler1.h>
#include <oz/generate.h>

namespace oz {

    template<typename T> struct BilateralFilter : public generator<T> {
        gpu_sampler<T,0> src_;
        float sigma_d_;
        float sigma_r_;
        float precision_;

        BilateralFilter( const gpu_image& src, float sigma_d, float sigma_r, float precision )
            : src_(src), sigma_d_(sigma_d), sigma_r_(sigma_r), precision_(precision) {}

        __device__ T operator()( int ix, int iy ) const {
            float twoSigmaD2 = 2.0f * sigma_d_ * sigma_d_;
            float twoSigmaR2 = 2.0f * sigma_r_ * sigma_r_;
            int halfWidth = int(ceilf( precision_ * sigma_d_ ));

            T c0 = src_(ix, iy);
            T sum = make_zero<T>();

            float norm = 0.0f;
            for ( int i = -halfWidth; i <= halfWidth; ++i ) {
                for ( int j = -halfWidth; j <= halfWidth; ++j ) {
                    float d = length(make_float2(i,j));

                    T c = src_(ix + i, iy + j);
                    T e = c - c0;

                    float kd = __expf( -dot(d,d) / twoSigmaD2 );
                    float kr = __expf( -dot(e,e) / twoSigmaR2 );

                    sum += kd * kr * c;
                    norm += kd * kr;
                }
            }
            sum /= norm;
            return sum;
        }
    };

    gpu_image bilateral_filter( const gpu_image& src, float sigma_d, float sigma_r, float precision ) {
        switch (src.format()) {
            case FMT_FLOAT:
                return generate(src.size(), BilateralFilter<float>(src, sigma_d, sigma_r, precision));
            case FMT_FLOAT3:
                return generate(src.size(), BilateralFilter<float3>(src, sigma_d, sigma_r, precision));
            default:
                OZ_INVALID_FORMAT();
        }
    }


    template<typename T, int dx, int dy> struct BilateralFilterXY : public generator<T> {
        gpu_sampler<T,0> src_;
        float sigma_d_;
        float sigma_r_;
        float precision_;

        BilateralFilterXY( const gpu_image& src, float sigma_d, float sigma_r, float precision )
            : src_(src), sigma_d_(sigma_d), sigma_r_(sigma_r), precision_(precision) {}

        __device__ T operator()( int ix, int iy ) const {
            float twoSigmaD2 = 2 * sigma_d_ * sigma_d_;
            float twoSigmaR2 = 2 * sigma_r_ * sigma_r_;
            int halfWidth = int(ceilf( precision_ * sigma_d_ ));

            T c0 = src_(ix, iy);
            T sum = make_zero<T>();

            float norm = 0;
            for ( int i = -halfWidth; i <= halfWidth; ++i ) {
                T c = src_(ix + dx * i, iy + dy * i);
                T e = c - c0;

                float kd = __expf( -i * i / twoSigmaD2 );
                float kr = __expf( -dot(e,e) / twoSigmaR2 );

                sum += kd * kr * c;
                norm += kd * kr;
            }
            sum /=  norm;
            return sum;
        }
    };

    gpu_image bilateral_filter_xy( const gpu_image& src, float sigma_d, float sigma_r, float precision ) {
        switch (src.format()) {
            case FMT_FLOAT:
                {
                    gpu_image tmp = generate(src.size(), BilateralFilterXY<float,1,0>(src, sigma_d, sigma_r, precision));
                    return generate(src.size(), BilateralFilterXY<float,0,1>(tmp, sigma_d, sigma_r, precision));
                }
            case FMT_FLOAT3:
                {
                    gpu_image tmp = generate(src.size(), BilateralFilterXY<float3,1,0>(src, sigma_d, sigma_r, precision));
                    return generate(src.size(), BilateralFilterXY<float3,0,1>(tmp, sigma_d, sigma_r, precision));
                }
            default:
                OZ_INVALID_FORMAT();
        }
    }

}
