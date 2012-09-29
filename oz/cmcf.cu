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
#include <oz/cmcf.h>
#include <oz/deriv2nd.h>
#include <oz/generate.h>
#include <oz/gpu_sampler7.h>

namespace oz {

    struct imp_cmcf : public generator<float3> {
        gpu_sampler<float3,0> src_;
        gpu_sampler<float2,1> tm_;
        gpu_sampler<float3,2> Ix_;
        gpu_sampler<float3,3> Iy_;
        gpu_sampler<float3,4> Ixx_;
        gpu_sampler<float3,5> Ixy_;
        gpu_sampler<float3,6> Iyy_;
        float step_;
        float weight_;

        imp_cmcf( const gpu_image& src, const gpu_image& tm,
                  const deriv2nd_t& d, float step, float weight)
                  : src_(src), tm_(tm),
                    Ix_(d.Ix), Iy_(d.Iy), Ixx_(d.Ixx), Ixy_(d.Ixy) , Iyy_(d.Iyy),
                    step_(step), weight_(weight) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            float3 val = src_(ix, iy);

            float3 Ix = Ix_(ix, iy);
            float3 Iy = Iy_(ix, iy);
            float3 Ixx = Ixx_(ix, iy);
            float3 Ixy = Ixy_(ix, iy);
            float3 Iyy = Iyy_(ix, iy);
            float2 t = tm_(ix, iy);
            float2 g = make_float2(t.y, -t.x);

            float3 diff_D = Ixx*Iy*Iy - 2*Ix*Iy*Ixy + Iyy*Ix*Ix;
            float3 diff_N = Ix*Ix + Iy*Iy;

            if (diff_N.x > 0) {
                float2 ng = normalize(make_float2(Ix.x, Iy.x));
                float speed = (1 - weight_) + weight_ * abs(dot(g, ng));
                val.x += step_ * speed * diff_D.x / diff_N.x;
            }
            if (diff_N.y > 0) {
                float2 ng = normalize(make_float2(Ix.y, Iy.y));
                float speed = (1 - weight_) + weight_ * abs(dot(g, ng));
                val.y += step_ * speed * diff_D.y / diff_N.y;
            }
            if (diff_N.z > 0) {
                float2 ng = normalize(make_float2(Ix.z, Iy.z));
                float speed = (1 - weight_) + weight_ * abs(dot(g, ng));
                val.z += step_ * speed * diff_D.z / diff_N.z;
            }

            return val;
        }
    };


    gpu_image cmcf( const gpu_image& src, const gpu_image& tm, float step, float weight ) {
        deriv2nd_t d = deriv2nd(src);
        return generate(src.size(), imp_cmcf(src, tm, d, step, weight));
    }

}


