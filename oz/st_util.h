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
#include <oz/st.h>

namespace oz {

    inline __host__ __device__ float3 st_normalized( float u, float v ) {
        float3 g = make_float3(u*u, v*v, u*v);
        float mag = sqrtf(g.x + g.y);
        if (mag > 0) g /= mag;
        return g;
    }

    inline __host__ __device__ float3 st_normalized( float3 u, float3 v ) {
        float3 x = st_normalized(u.x, v.x);
        float3 y = st_normalized(u.y, v.y);
        float3 z = st_normalized(u.z, v.z);
        return x + y + z;
    }

    inline __host__ __device__ float st2angle( const float3 g ) {
        return 0.5f * atan2(-2 * g.z, g.y - g.x);
    }

    inline __host__ __device__ float2 st2tangent( const float3 g ) {
        float phi = st2angle(g);
        return make_float2(cosf(phi), sinf(phi));
    }

    inline __host__ __device__ float2 st2gradient( const float3 g ) {
        float phi = 0.5f * atan2(2 * g.z, g.x - g.y);
        return make_float2(cosf(phi), sinf(phi));
    }

    inline __host__ __device__ float2 st2lambda( float3 g ) {
        float a = 0.5f * (g.y + g.x);
        float b = 0.5f * sqrtf(fmaxf(0.0f, g.y*g.y - 2*g.x*g.y + g.x*g.x + 4*g.z*g.z));
        return make_float2(a + b, a - b);
    }

    inline __host__ __device__ float4 st2tfm__( float3 g ) {
        float2 l = st2lambda(g);
        float2 t = st2tangent(g);
        return make_float4(t.x, t.y, l.x, l.y);
    }

    inline __host__ __device__ float tfm2A__( float4 t ) {
        float lambda1 = t.z;
        float lambda2 = t.w;
        return (lambda1 + lambda2 > 0)?
            (lambda1 - lambda2) / (lambda1 + lambda2) : 0;
    }

    inline __host__ __device__ float st2A( float3 g ) {
        float a = 0.5f * (g.y + g.x);
        float b = 0.5f * sqrtf(fmaxf(0.0f, g.y*g.y - 2*g.x*g.y + g.x*g.x + 4*g.z*g.z));
        float lambda1 = a + b;
        float lambda2 = a - b;
        return (lambda1 + lambda2 > 0)?
            (lambda1 - lambda2) / (lambda1 + lambda2) : 0;
    }

    inline __host__ __device__ float st2moa( float3 g, moa_t moa ) {
        float a = 0.5f * (g.y + g.x);
        float b = 0.5f * sqrtf(fmaxf(0.0f, g.y*g.y - 2*g.x*g.y + g.x*g.x + 4*g.z*g.z));
        float lambda1 = a + b;
        float lambda2 = a - b;
        switch (moa) {
            case MOA_SQUARED_EV:
                {
                    return (lambda1 + lambda2 > 0)?
                        (lambda1 - lambda2) / (lambda1 + lambda2) : 0;
                }
            case MOA_SQRT_EV:
                {
                    lambda1 = sqrtf(lambda1);
                    lambda2 = sqrtf(lambda2);
                    return (lambda1 + lambda2 > 0)?
                        (lambda1 - lambda2) / (lambda1 + lambda2) : 0;
                }
            case MOA_SQUARED_EV2:
                {
                    float A = (lambda1 + lambda2 > 0)?
                        (lambda1 - lambda2) / (lambda1 + lambda2) : 0;
                    return A*A;
                }
            default:
                break;
        }
        return 0;
    }

    inline __host__ __device__ float4 st2lfm( float3 g ) {
        float2 t = st2tangent(g);
        return make_float4( t.x, t.y, 1, 1 );
    }

    inline __host__ __device__ float4 st2lfm( float3 g, float alpha ) {
        float2 t = st2tangent(g);
        float A = st2A(g);
        return make_float4(
            t.x,
            t.y,
            ::clamp((alpha + A) / alpha, 0.1f, 2.0f),
            ::clamp(alpha / (alpha + A), 0.1f, 2.0f)
        );
    }

    inline __host__ __device__ float4 st2lfm( float3 g, moa_t moa, float alpha ) {
        float2 t = st2tangent(g);
        float A = st2moa(g, moa);
        return make_float4(
            t.x,
            t.y,
            ::clamp((alpha + A) / alpha, 0.1f, 2.0f),
            ::clamp(alpha / (alpha + A), 0.1f, 2.0f)
        );
    }

//     inline __host__ __device__ float3 st2tA( float3 g ) {
//         float2 t = st2tangent(g);
//         float A = st2A(g);
//         return make_float3( t.x, t.y, A );
//     }


    inline __host__ __device__
    void solve_eig_psd( float E, float F, float G, float& lambda1,
                        float& lambda2, float2& ev )
    {
        float B = (E + G) / 2;
        if (B > 0) {
            float D = (E - G) / 2;
            float FF = F*F;
            float R = sqrtf(D*D + FF);
            lambda1 = B + R;
            lambda2 = fmaxf(0, E*G - FF) / lambda1;

            if (R > 0) {
                if (D >= 0) {
                    float nx = D + R;
                    ev = make_float2(nx, F) * rsqrtf(nx*nx + FF);
                } else {
                    float ny = -D + R;
                    ev = make_float2(F, ny) * rsqrtf(FF + ny*ny);
                }
            } else {
                ev = make_float2(1, 0);
            }
        } else {
            lambda1 = lambda2 = 0;
            ev = make_float2(1, 0);
        }
    }


    inline __host__ __device__
    float2 solve_eig_psd_ev( float E, float F, float G )
    {
        float B = (E + G) / 2;
        if (B > 0) {
            float D = (E - G) / 2;
            float FF = F*F;
            float R = sqrtf(D*D + FF);

            if (R > 0) {
                if (D >= 0) {
                    float nx = D + R;
                    return make_float2(nx, F) * rsqrtf(nx*nx + FF);
                } else {
                    float ny = -D + R;
                    return make_float2(F, ny) * rsqrtf(FF + ny*ny);
                }
            }
        }
        return make_float2(1, 0);
    }


    inline __host__ __device__
    float solve_eig_psd_lambda1( float E, float F, float G ) {
        float B = (E + G) / 2;
        if (B > 0) {
            float D = (E - G) / 2;
            float FF = F*F;
            float R = sqrtf(D*D + FF);
            return B + R;
        }
        return 0;
    }


    inline __host__ __device__ float2 st_major_ev(const float4 g) {
        return solve_eig_psd_ev(g.x, g.z, g.y);
    }


    inline __host__ __device__ float2 st_minor_ev(const float4 g) {
        float2 ev = solve_eig_psd_ev(g.x, g.z, g.y);
        return make_float2(ev.y, -ev.x);
    }


    inline __host__ __device__ float st_lambda1(float4 g) {
        return solve_eig_psd_lambda1(g.x, g.z, g.y);
    }


    inline __host__ __device__ float3 st2tA( float3 g ) {
        float lambda1, lambda2;
        float2 ev;
        solve_eig_psd( g.x, g.z, g.y, lambda1, lambda2, ev);
        return make_float3(
            ev.y,
            -ev.x,
            (lambda1 + lambda2 > 0)? (lambda1 - lambda2) / (lambda1 + lambda2) : 0
        );
   }


    inline __host__ __device__ float3 st2tA( float3 g, float a_star ) {
        float lambda1, lambda2;
        float2 ev;
        solve_eig_psd( g.x, g.z, g.y, lambda1, lambda2, ev);
        float A = (lambda1 + lambda2 > 0)?
            (lambda1 - lambda2) / (lambda1 + lambda2) : 0;
        return make_float3(
            ev.y,
            -ev.x,
            (fmaxf(a_star, A) - a_star) / (1 - a_star)
        );
   }

}