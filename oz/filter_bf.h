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

    template<typename T> T make_error_color();
    template<> inline __host__ __device__ float  make_error_color() { return 0; }
    template<> inline __host__ __device__ float3 make_error_color() { return make_float3(53,80,67); }


    template<typename T, typename SRC>
    class filter_bf {
    public:
         __host__ __device__ filter_bf( const SRC& src, T c0, float sigma_d,
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

        __host__ __device__ void operator()( float u, float2 p ) {
            T c1 = src_(p.x, p.y);
            T r = c1 - c0_;
            float kd = expf(-u * u / twoSigmaD2_);
            float kr = expf(-dot(r,r) / twoSigmaR2_);
            c_ += kd * kr * c1;
            w_ += kd * kr;
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


    template<typename T, typename SRC, bool src_linear>
    class filter_bf_trapez {
    public:
         __host__ __device__ filter_bf_trapez( const SRC& src, T c0, float sigma_d,
                                               float sigma_r, float precision ) : src_(src)
         {
            radius_ = precision * sigma_d;
            twoSigmaD2_ = 2 * sigma_d * sigma_d;
            twoSigmaR2_ = 2 * sigma_r * sigma_r;
            c0_ = c0;
            c_ = make_zero<T>();
            w_ = u_ = 0;
        }

        __host__ __device__ float radius() const {
            return radius_;
        }

        inline __host__ __device__ T result() const {
            return (w_ > 0)? c_ / w_ : c0_;
        }

        __host__ __device__ void operator()( float u, float2 p ) {
            if (u == 0) {
                u_ = 0;
                cp_ = c0_;
                wp_ = 1;
            } else {
                float du = fabsf(u - u_);
                u_ = u;

                c_ += cp_ * du / 2;
                w_ += wp_ * du / 2;

                float kd = expf(-u * u / twoSigmaD2_);
                if (src_linear) {
                    T c1 = src_(p.x, p.y);
                    float kr = expf(-squared(c1 - c0_) / twoSigmaR2_);
                    cp_ = kd * kr * c1;
                    wp_ = kd * kr;
                } else {
                    p -= make_float2(0.5f, 0.5f);
                    float2 ip = floor(p);
                    float2 fp = p - ip;

                    T c1, c2;
                    float f;
                    if (fp.x > 1e-4f) {
                        float2 q = make_float2(ip.x, p.y);
                        c1 = src_(q.x, q.y);
                        c2 = src_(q.x + 1, q.y);
                        f = fp.x;
                    } else if (fp.y > 1e-4f) {
                        float2 q = make_float2(p.x, ip.y);
                        c1 = src_(q.x, q.y);
                        c2 = src_(q.x, q.y + 1);
                        f = fp.y;
                    } else {
                        c1 = c2 = src_(p);
                        f = 0;
                    }

                    float kr1 = (1 -  f) * __expf( -squared(c1 - c0_) / twoSigmaR2_ );
                    float kr2 = f * __expf( -squared(c2 - c0_) / twoSigmaR2_ );

                    cp_ = kd * (kr1 * c1 + kr2 * c2);
                    wp_ = kd * (kr1 + kr2);

                }

                c_ += cp_ * du / 2;
                w_ += wp_ * du / 2;
            }
        }

        inline __host__ __device__ void error() {
            c_ = make_error_color<T>();
            w_ = 1;
        }

    private:
        const SRC& src_;
        float radius_;
        float twoSigmaD2_;
        float twoSigmaR2_;
        T c0_;
        T cp_;
        T c_;
        float wp_;
        float w_;
        float u_;
    };

}
