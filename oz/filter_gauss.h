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
    class filter_gauss_1d {
    public:
        inline __host__ __device__ filter_gauss_1d( const SRC& src, float sigma, float precision=2 )
            : src_(src)
        {
            radius_ = precision * sigma;
            twoSigma2_ = 2 * sigma * sigma;
            c_ = make_zero<T>();
            w_ = 0;
        }

        inline __host__ __device__ float radius() const {
            return radius_;
        }

        inline __host__ __device__ T result() const {
            return c_ / w_;
        }

        inline __host__ __device__ void operator()( float u, float2 p ) {
            float k = expf(-u * u / twoSigma2_);
            c_ += k * src_(p.x, p.y);
            w_ += k;
        }

    private:
        const SRC& src_;
        float radius_;
        float twoSigma2_;
        T c_;
        float w_;
    };

}
