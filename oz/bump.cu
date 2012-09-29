//
// by Daniel Müller and Jan Eric Kyprianidis
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
#include <oz/bump.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>


namespace oz {
    struct BumpPhong : public generator<float> {
        gpu_sampler<float,0> src_;
        float scale_;
        float specular_;
        float shininess_;

        BumpPhong( const gpu_image& src, float scale, float specular, float shininess )
            : src_(src), scale_(scale), specular_(specular), shininess_(shininess) {}

        inline __device__ float operator()( int ix, int iy ) const {
            const float x = src_(ix, iy) * scale_;

            const float a = src_(ix - 1, iy - 1) * scale_;
            const float b = src_(ix    , iy - 1) * scale_;
            const float c = src_(ix + 1, iy - 1) * scale_;
            const float d = src_(ix - 1, iy    ) * scale_;
            const float e = src_(ix + 1, iy    ) * scale_;
            const float f = src_(ix - 1, iy + 1) * scale_;
            const float g = src_(ix    , iy + 1) * scale_;
            const float h = src_(ix + 1, iy + 1) * scale_;

            float3 N = make_float3(0.f);
            const float s = 1.f;

            N += cross(make_float3( s, 0, x - d), make_float3( 0, s, a - d));
            N += cross(make_float3(-s, 0, a - b), make_float3( 0,-s, x - b));
            N += cross(make_float3( 0,-s, x - b), make_float3( s, 0, c - b));
            N += cross(make_float3( 0, s, c - e), make_float3(-s, 0, x - e));
            N += cross(make_float3(-s, 0, x - e), make_float3( 0,-s, h - e));
            N += cross(make_float3( 1, 0, h - g), make_float3( 0, s, x - g));
            N += cross(make_float3( 0, s, x - g), make_float3(-s, 0, f - g));
            N += cross(make_float3( 0,-s, f - d), make_float3( s, 0, x - d));

            N = normalize(N);

            const float3 L = normalize(make_float3(-1.0f, 1.0f, 1.0f));
            const float NDotL = dot(N, L);

            return powf(NDotL, shininess_) * specular_;
        }
    };
}

oz::gpu_image oz::bump_phong( const gpu_image& src, const float scale,
                             const float specular, const float shininess)
{
    return generate(src.size(), BumpPhong(src, scale, specular, shininess));
}
