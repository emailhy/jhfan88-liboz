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
#include <oz/xdog.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>
#include <oz/transform.h>


namespace oz {
    struct imp_xdog_sharpen : generator<float> {
        gpu_sampler<float,0> src_;
        float sigma_;
        float k_;
        float p_;
        float precision_;

        imp_xdog_sharpen( const gpu_image& src, float sigma, float k, float p, float precision )
            : src_(src), sigma_(sigma), k_(k), p_(p), precision_(precision) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float twoSigmaE2 = 2 * sigma_ * sigma_;
            float twoSigmaR2 = twoSigmaE2 * k_ * k_;
            int halfWidth = int(ceilf( precision_ * sigma_ * k_ ));

            float gE = 0;
            float gR = 0;
            float2 norm = make_float2(0);

            for ( int i = -halfWidth; i <= halfWidth; ++i ) {
                for ( int j = -halfWidth; j <= halfWidth; ++j ) {
                    float d = length(make_float2(i,j));
                    float kE = __expf(-d *d / twoSigmaE2);
                    float kR = __expf(-d *d / twoSigmaR2);

                    float c = src_(ix + i, iy + j);
                    gE += kE * c;
                    gR += kR * c;
                    norm += make_float2(kE, kR);
                }
            }

            gE /= norm.x;
            gR /= norm.y;

            return (1 + p_) * gE - p_ * gR;
        }
    };


    struct imp_xdog_levels : unary_function<float,float> {
        float lambda1_;
        float lambda2_;
        float gamma_;

        imp_xdog_levels( float lambda1, float lambda2, float gamma )
            : lambda1_(lambda1), lambda2_(lambda2), gamma_(gamma) {}

        inline __device__ float operator()( float H ) const {
            float x = clamp((H - lambda1_)/(lambda2_ - lambda1_), 0.0f, 1.0f);
            return __powf(x, gamma_);
        }
    };
}


oz::gpu_image oz::xdog_sharpen( const gpu_image& src, float sigma, float k, float p, float precision ) {
    return generate(src.size(), imp_xdog_sharpen(src, sigma, k, p, precision));
}



oz::gpu_image oz::xdog_levels( const gpu_image& src, float lambda1, float lambda2, float gamma )
{
    return transform(src, imp_xdog_levels(lambda1, lambda2, gamma));
}
