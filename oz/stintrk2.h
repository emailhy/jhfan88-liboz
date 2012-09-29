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
#include <oz/st_util.h>

namespace oz {

    template <class ST, class F, int order, bool adaptive>
    inline __host__ __device__ void st3_int( float2 p0, const ST& st, F& f,
                                             unsigned w, unsigned h, float step_size )
    {
        f(0, p0);
        float2 v0 = st2tangent(st(p0));
        float sign = -1;
        float dr = f.radius() / CUDART_PI_F;
        do {
            float2 v = v0 * sign;
            float2 p = p0;
            float u = 0;

            for (;;) {
                float2 t = st2tangent(st(p));
                if (order == 2) {
                    if (dot(v, t) < 0) t = -t;
                    t = st2tangent(st(p + 0.5f * step_size * t));
                }
                float vt = dot(v, t);
                if (vt < 0) {
                    t = -t;
                    vt = -vt;
                }

                v = t;
                p += step_size * t;
                if (adaptive) {
                    float Lk = dr * acosf(fminf(vt,1));
                    u += fmaxf(step_size, Lk);
                } else {
                    u += step_size;
                }

                if ((u >= f.radius()) ||
                    (p.x < 0) || (p.x >= w) || (p.y < 0) || (p.y >= h)) break;

                f(copysignf(u, sign), p);
            }

            sign *= -1;
        } while (sign > 0);
    }


    template <class ST, class F, int order>
    inline __host__ __device__ void st3_int_ustep( float2 p0, const ST& st, F& f,
                                                   unsigned w, unsigned h, float step_size )
    {
        float2 v0 = st2tangent(st(p0));
        float sign = -1;
        float dr = f.radius() / CUDART_PI_F;
        do {
            float2 v = v0 * sign;
            float2 p = p0;
            float u = 0;
            f(0, p0);
            for (;;) {
                float2 t = st2tangent(st(p));
                if (order == 2) {
                    if (dot(v, t) < 0) t = -t;
                    t = st2tangent(st(p + 0.5f * step_size * t));
                }
                float vt = dot(v, t);
                if (vt < 0) {
                    t = -t;
                    vt = -vt;
                } else if (vt == 0) break;

                v = t;
                p += step_size * t;
                u += step_size;

                float2 fp = make_float2(fract(p.x + 0.5f), fract(p.y + 0.5f));
                float du;
                if ((fp.x == 0) || (fp.y == 0)) du = 0;
                else {
                    du = CUDART_NORM_HUGE_F;

                    if (t.x > 0)
                        du = fp.x / t.x;
                    else if (t.x < 0)
                        du = (fp.x - 1) / t.x;

                    if (t.y > 0)
                        du = fminf(du, fp.y / t.y);
                    else if (t.y < 0)
                        du = fminf(du, (fp.y - 1) / t.y);
                }

                if (du < step_size) {
                    float2 q = p - t * du;
                    float qu = u - du;

                    if ((qu >= f.radius()) ||
                        (q.x < 0) || (q.x >= w) || (q.y < 0) || (q.y >= h)) break;

                    f(copysignf(qu, sign), q);
                }
            }
            sign *= -1;
        } while (sign > 0);
    }

}
