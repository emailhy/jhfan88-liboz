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

    inline __device__ float srgb2linear( float x ) {
        return ( x > 0.04045f ) ? __powf( ( x + 0.055f ) / 1.055f, 2.4f ) : x / 12.92f;
    }


    inline __device__ float linear2srgb( float x ) {
        return ( x > 0.0031308f ) ? (( 1.055f * __powf( x, ( 1.0f / 2.4f ))) - 0.055f ) : 12.92f * x;
    }


    inline __device__ float sbgr2linear( float x ) { return srgb2linear(x); }
    inline __device__ float3 sbgr2linear( float3 sbgr ) {
        return make_float3(srgb2linear(sbgr.x), srgb2linear(sbgr.y), srgb2linear(sbgr.z));
    }


    inline __device__ float linear2sbgr( float x ) { return linear2srgb(x); }
    inline __device__ float3 linear2sbgr( float3 bgr ) {
        return make_float3(linear2srgb(bgr.x), linear2srgb(bgr.y), linear2srgb(bgr.z));
    }


    inline __device__ float3 bgr2xyz( float3 bgr ) {
        float b = bgr.x;
        float g = bgr.y;
        float r = bgr.z;
        float x = 0.412453f * r + 0.357580f * g + 0.180423f * b;
        float y = 0.212671f * r + 0.715160f * g + 0.072169f * b;
        float z = 0.019334f * r + 0.119193f * g + 0.950227f * b;
        return make_float3(x, y, z);
    }


    inline __device__ float3 xyz2bgr( float3 xyz ) {
        float x = xyz.x;
        float y = xyz.y;
        float z = xyz.z;
        float r =  3.240479f * x - 1.537150f * y - 0.498535f * z;
        float g = -0.969256f * x + 1.875991f * y + 0.041556f * z;
        float b =  0.055648f * x - 0.204043f * y + 1.057311f * z;
        return make_float3(b, g, r);
    }


    inline __device__ float3 xyz2lab( float3 xyz ) {
        float x = xyz.x / 0.950456f;
        float y = xyz.y;
        float z = xyz.z / 1.088754f;

        float fx = ( x > 0.008856f )? cbrtf(x) : 7.787f*x + 16.0f/116.0f;
        float fy = ( y > 0.008856f )? cbrtf(y) : 7.787f*y + 16.0f/116.0f;
        float fz = ( z > 0.008856f )? cbrtf(z) : 7.787f*z + 16.0f/116.0f;

        float L = 116 * fy - 16;
        float a = 500 * ( fx - fy );
        float b = 200 * ( fy - fz );

        return make_float3(L, a, b);
    }


    inline __device__ float3 lab2xyz( float3 lab ) {
        float L = lab.x;
        float a = lab.y;
        float b = lab.z;

        float fy = ( L + 16.0f ) / 116.0f;
        float fx = fy + a / 500.0f;
        float fz = fy - b / 200.0f;

        float x = 0.950456f * ((fx > 0.206897f)? fx*fx*fx : (fx - 16.0f/116.0f) / 7.787f);
        float y =             ((fy > 0.206897f)? fy*fy*fy : (fy - 16.0f/116.0f) / 7.787f);
        float z = 1.088754f * ((fz > 0.206897f)? fz*fz*fz : (fz - 16.0f/116.0f) / 7.787f);

        return make_float3(x, y, z);
    }


    inline __device__ float3 xyz2luv( float3 xyz ) {
        //const float xn = 0.312713f;
        //const float yn = 0.329016f;
        //const float un = 4 * xn / ( -2 * xn + 12 * yn + 3 );
        //const float vn = 9 * yn / ( -2 * xn + 12 * yn + 3 );
        const float un = 0.19793943f;
        const float vn = 0.46831096f;
        const float T29o3 = 24389.0f/27.0f; // (29/3)^3
        const float T6o29 = 216.0f/24389.0f; // (6/29)^3

        float X = xyz.x;
        float Y = xyz.y;
        float Z = xyz.z;

        float u = 4 * X / (X + 15*Y + 3*Z);
        float v = 9 * Y / (X + 15*Y + 3*Z);

        float L = ( Y <= T6o29 ) ? T29o3 * Y : 116 * cbrtf(Y) - 16;
        float U = 13 * L * (u - un);
        float V = 13 * L * (v - vn);

        return make_float3(L, U, V);
    }


    inline __device__ float3 luv2xyz( float3 luv ) {
        const float un = 0.19793943f;
        const float vn = 0.46831096f;
        const float T3o29 = 27.0f/24389.0f; // (3/29)^3

        float L = luv.x;
        float U = luv.y;
        float V = luv.z;
        if (L <= 0) return make_float3(0);

        float u = U / (13*L) + un;
        float v = V / (13*L) + vn;

        float Y = (L <= 8)? T3o29*L : __powf((L + 16)/116, 3);
        float X = Y * 9*u / (4*v);
        float Z = Y * (12 - 3*u - 20*v) / (4*v);

        return make_float3(X, Y, Z);
    }


    inline __device__ float luv2nvac( float3 luv ) {
        float L = luv.x;
        if (L <= 0) return 0;

        float u = luv.y / (13*L);
        float v = luv.z / (13*L);

        const float L_a =  20;
        const float K_Br = 0.2717f * ( 6.469f + 6.362f * powf(L_a, 0.4495f) ) /
                                     ( 6.469f + powf(L_a, 0.4495f) );

        float s_uv = 13 * sqrtf(u*u + v*v);
        float theta = atan2(v, u);
        float q = -0.01585f
                  -0.03016f * __cosf(theta)   - 0.04556f * __cosf(2*theta)
                  -0.02667f * __cosf(3*theta) - 0.00295f * __cosf(4*theta)
                  +0.14592f * __sinf(theta)   + 0.05084f * __sinf(2*theta)
                  -0.01900f * __sinf(3*theta) - 0.00764f * __sinf(4*theta);

        return L * (1 + ( -0.1340f * q + 0.0872f * K_Br ) * s_uv);
    }

}