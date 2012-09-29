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
#include <oz/gauss.h>
#include <oz/gpu_sampler1.h>
#include <oz/generate.h>

namespace oz {

    template<typename T> struct GaussFilter : public generator<T> {
        gpu_sampler<T,0> src_;
        float sigma_;
        float precision_;

        GaussFilter( const gpu_image& src, float sigma, float precision )
            : src_(src), sigma_(sigma), precision_(precision) {}

        __device__ T operator()( int ix, int iy ) const {
            float twoSigma2 = 2.0f * sigma_ * sigma_;
            int halfWidth = int(ceilf( precision_ * sigma_ ));

            T sum = make_zero<T>();
            float norm = 0;
            for ( int i = -halfWidth; i <= halfWidth; ++i ) {
                for ( int j = -halfWidth; j <= halfWidth; ++j ) {
                    float d = length(make_float2(i,j));
                    float kernel = __expf( -d *d / twoSigma2 );
                    T c = src_(ix + i, iy + j);
                    sum += kernel * c;
                    norm += kernel;
                }
            }
            sum /=  norm;
            return sum;
        }
    };

    gpu_image gauss_filter( const gpu_image& src, float sigma, float precision ) {
        if (sigma <= 0) return src;
        switch (src.format()) {
            case FMT_FLOAT:  return generate(src.size(), GaussFilter<float>(src, sigma, precision));
            case FMT_FLOAT2: return generate(src.size(), GaussFilter<float2>(src, sigma, precision));
            case FMT_FLOAT3: return generate(src.size(), GaussFilter<float3>(src, sigma, precision));
            case FMT_FLOAT4: return generate(src.size(), GaussFilter<float4>(src, sigma, precision));
            default:
                OZ_INVALID_FORMAT();
        }
    }


    template<typename T, int dx, int dy> struct GaussFilterXY : public generator<T> {
        gpu_sampler<T,0> src_;
        float sigma_;
        float precision_;

        GaussFilterXY( const gpu_image& src, float sigma, float precision )
            : src_(src), sigma_(sigma), precision_(precision) {}

        __device__ T operator()( int ix, int iy ) const {
            float twoSigma2 = 2.0f * sigma_ * sigma_;
            int halfWidth = ceilf( precision_ * sigma_ );

            T sum = src_(ix, iy);
            float norm = 1;
            for ( int i = 1; i <= halfWidth; ++i ) {
                float kernel = __expf( -i *i / twoSigma2 );
                sum += kernel * (src_(ix + dx * i, iy + dy * i) + src_(ix - dx * i, iy - dy * i));
                norm += 2 * kernel;
            }
            sum /=  norm;
            return sum;
        }
    };

    template<typename T>
    gpu_image gauss_filter_xyT( const gpu_image& src, float sigma, float precision ) {
        gpu_image tmp = generate(src.size(), GaussFilterXY<T,1,0>(src, sigma, precision));
        return generate(tmp.size(), GaussFilterXY<T,0,1>(tmp, sigma, precision));
    }

    gpu_image gauss_filter_xy( const gpu_image& src, float sigma, float precision ) {
        if (sigma <= 0) return src;
        switch (src.format()) {
            case FMT_FLOAT:  return gauss_filter_xyT<float >(src, sigma, precision);
            case FMT_FLOAT2: return gauss_filter_xyT<float2>(src, sigma, precision);
            case FMT_FLOAT3: return gauss_filter_xyT<float3>(src, sigma, precision);
            case FMT_FLOAT4: return gauss_filter_xyT<float4>(src, sigma, precision);
            default:
                OZ_INVALID_FORMAT();
        }
    }


    template<typename T> struct GaussFilter3x3 : public generator<T> {
        gpu_sampler<T,0> src_;

        GaussFilter3x3( const gpu_image& src ): src_(src) {}

        // [0.216, 0.568, 0.216], sigma ~= 0.680
        __device__ T operator()( int ix, int iy ) const {
            return (
                0.046656f * src_(ix-1, iy-1) +
                0.122688f * src_(ix,   iy-1) +
                0.046656f * src_(ix+1, iy-1) +
                0.122688f * src_(ix-1, iy) +
                0.322624f * src_(ix,   iy) +
                0.122688f * src_(ix+1, iy) +
                0.046656f * src_(ix-1, iy+1) +
                0.122688f * src_(ix,   iy+1) +
                0.046656f * src_(ix+1, iy+1)
            );
        }
    };

    gpu_image gauss_filter_3x3( const gpu_image& src ) {
        switch (src.format()) {
            case FMT_FLOAT:  return generate(src.size(), GaussFilter3x3<float >(src));
            case FMT_FLOAT2: return generate(src.size(), GaussFilter3x3<float2>(src));
            case FMT_FLOAT3: return generate(src.size(), GaussFilter3x3<float3>(src));
            case FMT_FLOAT4: return generate(src.size(), GaussFilter3x3<float4>(src));
            default:
                OZ_INVALID_FORMAT();
        }
    }


    template<typename T> struct GaussFilter5x5 : public generator<T> {
        gpu_sampler<T,0> src_;

        GaussFilter5x5( const gpu_image& src ): src_(src) {}

        // [0.03134, 0.24, 0.45732, 0.24, 0.03134], sigma ~= 0.867
        __device__ T operator()( int ix, int iy ) const {
            const float kernel[5][5] = {
                { 0.0009821956f, 0.0075216f, 0.0143324088f, 0.0075216f, 0.0009821956 },
                { 0.0075216f,    0.0576f,    0.1097568f,    0.0576f,    0.0075216 },
                { 0.0143324088f, 0.1097568f, 0.2091415824f, 0.1097568f, 0.0143324088 },
                { 0.0075216f,    0.0576f,    0.1097568f,    0.0576f,    0.0075216 },
                { 0.0009821956f, 0.0075216f, 0.0143324088f, 0.0075216f, 0.0009821956 }
            };
            T sum = make_zero<T>();
            for ( int j = 0; j < 5; ++j ) {
                for ( int i = 0; i < 5; ++i ) {
                    T c = src_(ix + i -2, iy + j - 2);
                    sum += kernel[j][i] * c;
                }
            }
            return sum;
        }
    };

    gpu_image gauss_filter_5x5( const gpu_image& src ) {
        switch (src.format()) {
            case FMT_FLOAT:  return generate(src.size(), GaussFilter5x5<float >(src));
            case FMT_FLOAT2: return generate(src.size(), GaussFilter5x5<float2>(src));
            case FMT_FLOAT3: return generate(src.size(), GaussFilter5x5<float3>(src));
            case FMT_FLOAT4: return generate(src.size(), GaussFilter5x5<float4>(src));
            default:
                OZ_INVALID_FORMAT();
        }
    }

}