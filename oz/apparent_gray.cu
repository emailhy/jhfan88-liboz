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
#include <oz/apparent_gray.h>
#include <oz/color.h>
#include <oz/pyr.h>
#include <oz/make.h>
#include <oz/shuffle.h>
#include <oz/transform.h>
#include <vector>

namespace oz {
    struct ApparentGraySharpen : public binary_function<float4,float4,float4> {
        float k_, p_;

        ApparentGraySharpen( float k, float p ) : k_(k), p_(p) {}

        __device__ float4 operator()( float4 l, float4 h ) const {
            float deltaE = sqrtf(fabs(h.x*h.x + h.y*h.y + h.z*h.z));
            float lambda;
            if (fabs(h.w) > 1)
                lambda = k_ * __powf(deltaE/fabs(h.w), p_);
            else
                lambda = 0;

            return make_float4(make_float3(l) + make_float3(h),
                               l.w + (1+lambda) * h.w );
        }
    };


    struct ApparentGrayWeight : public binary_function<float4,float4,float> {
        float k_, p_;

        ApparentGrayWeight( float k, float p ) : k_(k), p_(p) {}

        __device__ float operator()( float4 l, float4 h ) const {
            float deltaE = sqrtf(fabs(h.x*h.x + h.y*h.y + h.z*h.z));
            float lambda;
            if (fabs(h.w) > 1)
                lambda = k_ * __powf(deltaE/fabs(h.w), p_);
            else
                lambda = 0;
            return lambda;
        }
    };
}


oz::gpu_image oz::apparent_gray_sharpen( const gpu_image& lp, const gpu_image& hp, float k, float p ) {
    return transform(lp, hp, ApparentGraySharpen(k, p));
}


oz::gpu_image oz::apparent_gray_weight( const gpu_image& lp, const gpu_image& hp, float k, float p ) {
    return transform(lp, hp, ApparentGrayWeight(k, p));
}


oz::gpu_image oz::apparent_gray( const gpu_image& src, int N, float* k, float p ) {
    gpu_image img = make(rgb2lab(src), rgb2nvac(src));
    std::vector<gpu_image> LP;
    for (int i = 0; i < N; ++i) {
        gpu_image dw = pyrdown_gauss5x5(img);
        gpu_image up = pyrup_gauss5x5(dw, img.w(), img.h());
        gpu_image hp = img - up;
        LP.push_back(hp);
        img = dw;
    }

    for (int i = N-1; i >= 0; --i) {
        gpu_image up = pyrup_gauss5x5(img, LP[i].w(), LP[i].h());
        img = apparent_gray_sharpen(up, LP[i], k[i], p);
    }

    return shuffle(img, 3);
}
