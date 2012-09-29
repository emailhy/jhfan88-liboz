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
#include <oz/cpu_image.h>
#include <vector>

namespace oz {

    template <typename TF, typename F, bool midpoint>
    inline __host__ __device__ void lic_integrate( float2 p0, const TF& tf, F& f,
                                                   unsigned w_, unsigned h_ )
    {
        float2 v0 = tf(p0.x, p0.y);
        float sign = -1;
        do {
            float2 v = v0 * sign;
            float2 p = p0;
            float u = 0;

            for (;;) {
                float2 t = tf(p.x, p.y);
                float vt = dot(v, t);
                if (vt <= 0) {
                    if (vt == 0) break;
                    t = -t;
                    vt = -vt;
                }

                float2 fp = make_float2( fract(p.x), fract(p.y) );
                float hmin = CUDART_NORM_HUGE_F;
                if (fp.x > 0) {
                    if (t.x < 0) {
                        hmin = -fp.x / t.x;
                    } else if (t.x > 0) {
                        hmin = (1 - fp.x) / t.x;
                    }
                }

                if (fp.y > 0) {
                    if (t.y < 0) {
                        hmin = fminf(hmin, -fp.y / t.y);
                    } else if (t.y > 0) {
                        hmin = fminf(hmin, (1 - fp.y) / t.y);
                    }
                }

                float2 dp = t * (hmin + 0.025f);
                float du = length(dp);

                if (!midpoint)
                    f(copysignf(u, sign), du, p);
                else
                    f(copysignf(u + 0.5f * du, sign), du, p + 0.5f * dp);

                p += dp;
                u += du;
                v = t;

                if ((u >= f.radius()) ||
                    (p.x < 0) || (p.x >= w_) || (p.y < 0) || (p.y >= h_)) break;
            }

            sign *= -1;
        } while (sign > 0);
    }

    OZAPI gpu_image licint_gauss_flt( const gpu_image& src, const gpu_image& tf, float sigma, float precision=2 );
    OZAPI gpu_image licint_bf_flt( const gpu_image& src, const gpu_image& tf, float sigma_d, float sigma_r, float precision=2 );
    OZAPI std::vector<float3> licint_path( int ix, int iy, const cpu_image& vf, float sigma, float precision, bool midpoint );

}
