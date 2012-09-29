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

   inline __host__ __device__ float3 colormap_H(float p) {
        int i = (int)(360 * fract(p));
        int k = i / 60;
        float m = 1.0f * (i % 60) / 59.0f;
        switch (k) {
            case 0: return make_float3(  0,   m,   1);
            case 1: return make_float3(  0,   1, 1-m);
            case 2: return make_float3(  m,   1,   0);
            case 3: return make_float3(  1, 1-m,   0);
            case 4: return make_float3(  1,   0,   m);
            case 5: return make_float3(1-m,   0,   1);

        }
        return make_float3(0);
    }

    inline __host__ __device__ float3 colormap_jet2(float p) {
        int i = (int)(299 * ::clamp(p, 0.0f, 1.0f));
        int k = i / 60;
        float m = 1.0f * (i % 60) / 60.0f;
        switch (k) {
            case 0: return make_float3(  1,   m,   0);
            case 1: return make_float3(1-m,   1,   0);
            case 2: return make_float3(  0,   1,   m);
            case 3: return make_float3(  m, 1-m,   1);
            case 4: return make_float3(1-m,   0,   1);
        }
        return make_float3(0);
    }


    inline __host__ __device__ float3 colormap_jet(float p) {
        float i = clamp(p, 0.0f, 1.0f);
        if (i <= 0.125f)
            return lerp(make_float3(0.5608f,0,0), make_float3(1,0,0), i / 0.125f);
        i -= 0.125f;
        if (i <= 0.25f)
            return lerp(make_float3(1,0,0), make_float3(1,1,0), i / 0.25f);
        i -= 0.25f;
        if (i <= 0.25f)
            return lerp(make_float3(1,1,0), make_float3(0,1,1), i / 0.25f);
        i -= 0.25f;
        if (i <= 0.25f)
            return lerp(make_float3(0,1,1), make_float3(0,0,1), i / 0.25f);
        i -= 0.25f;
        return lerp(make_float3(0,0,1), make_float3(0,0,0.5f), i / 0.125f);
    }

}
