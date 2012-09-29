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
#include <oz/spatial_color2gray.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>


namespace {

    __device__ float kstep(float x, float K, float B1, float B2) {
        if (x < B1) return K;
        if (x > B2) return 0;
        return K - (x - B1) / (B2 - B1);
    }

    struct SpatialColor2gray : public oz::generator<float>{
        gpu_sampler<float3,0> src_;
        int radius_;
        float K_, B1_, B2_;

        SpatialColor2gray( const oz::gpu_image& src, int radius, float K, float B1, float B2 )
            : src_(src), radius_(radius), K_(K), B1_(B1), B2_(B2) {}

        __device__ float operator()( int ix, int iy ) const {
            float3 sum = make_float3(0);
            for (int j = -radius_; j <= radius_; ++j) {
                for (int i = -radius_; i <= radius_; ++i) {
                    sum += src_(ix + i, iy + j);
                }
            }

            float3 lab = src_(ix, iy);
            int d = 2 * radius_ + 1;
            float3 hp = lab - sum / (d*d);
            float chp = sqrtf( hp.y*hp.y + hp.z*hp.z );

            return lab.x + sign(hp.x) * kstep(fabs(hp.x), K_, B1_, B2_) * chp;
        }
    };

}


oz::gpu_image oz::spatial_color2gray( const gpu_image& src, int radius, float K, float B1, float B2 ) {
    return generate(src.size(), SpatialColor2gray(src, radius, K, B1, B2));
}
