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
#pragma once

#include <oz/math_util.h>

namespace oz {

    template <typename T, typename SRC>
    class filter_erf_gauss {
    public:
        inline __host__ __device__ filter_erf_gauss( const SRC& src, float sigma, float precision )
            : src_(src)
        {
            radius_ = precision * sigma;
            sigmaSqrtTwo_ = sigma * CUDART_SQRT_TWO_F;
            c_ = make_zero<T>();
            w_ = 0;
        }

        inline __host__ __device__ float radius() const {
            return radius_;
        }

        inline __host__ __device__ T result() const {
            return c_ / w_;
        }

        inline __host__ __device__ void operator()( float u, float du, float2 p ) {
            float k = erff((u+du) / sigmaSqrtTwo_) - erff((u) / sigmaSqrtTwo_);
            c_ += k * src_(p.x, p.y);
            w_ += k;
        }

    private:
        const SRC& src_;
        float radius_;
        float sigmaSqrtTwo_;
        T c_;
        float w_;
    };


    template<typename T, typename SRC, bool ustep = false>
    class filter_erf_bf {
    public:
         __host__ __device__ filter_erf_bf( const SRC& src, T c0, float sigma_d,
                                            float sigma_r, float precision )
             : src_(src)
         {
            radius_ = precision * sigma_d;
            twoSigmaD2_ = 2 * sigma_d * sigma_d;
            twoSigmaR2_ = 2 * sigma_r * sigma_r;
            c0_ = c0;
            c_ = make_zero<T>();
            w_ = 0;
        }

        __host__ __device__ float radius() const {
            return radius_;
        }

        inline __host__ __device__ T result() const {
            return c_ / w_;
        }

        __host__ __device__ void operator()( float u, float du, float2 p ) {
            if (!ustep) {
                T c1 = src_(p.x, p.y);
                T r = c1 - c0_;
                float kd = expf(-u * u / twoSigmaD2_);
                float kr = expf(-dot(r,r) / twoSigmaR2_);
                c_ += kd * kr * c1;
                w_ += kd * kr;
            } else {
                float2 dp = 0.5f * make_float2((float)(fabs(fract(p.y - 0.5f)) < 1e-5f),
                                               (float)(fabs(fract(p.x - 0.5f)) < 1e-5f));
                T c1 = src_(p + dp);
                T c2 = src_(p - dp);
                T r1 = c1 - c0_;
                T r2 = c2 - c0_;
                float kd = expf(-u * u / twoSigmaD2_);
                float kr1 = expf(-dot(r1,r1) / twoSigmaR2_);
                float kr2 = expf(-dot(r2,r2) / twoSigmaR2_);
                c_ += kd * (kr1 * c1 + kr2 * c2);
                w_ += kd * (kr1 + kr2);
            }
        }

    private:
        const SRC& src_;
        float radius_;
        float twoSigmaD2_;
        float twoSigmaR2_;
        T c0_;
        T c_;
        float w_;
    };

}
