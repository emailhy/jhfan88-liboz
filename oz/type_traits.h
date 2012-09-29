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

    template<typename T> struct type_traits {
        enum { N = 1 };
        static image_format_t format() { return FMT_STRUCT; }
        typedef T value_type;
        typedef T scalar_type;
        typedef T texture_type;
    };

    template<> struct type_traits<uchar> {
        enum { N = 1 };
        static image_format_t format() { return FMT_UCHAR; }
        typedef uchar value_type;
        typedef uchar scalar_type;
        typedef uchar texture_type;

        static inline __host__ __device__ uchar x( uchar s ) { return s; }
        static inline __host__ __device__ uchar y( uchar s ) { return s; }
        static inline __host__ __device__ uchar z( uchar s ) { return s; }
        static inline __host__ __device__ uchar w( uchar s ) { return 255; }
    };

    template<> struct type_traits<uchar2> {
        enum { N = 2 };
        static image_format_t format() { return FMT_UCHAR2; }
        typedef uchar2 value_type;
        typedef uchar scalar_type;
        typedef uchar2 texture_type;

        static inline __host__ __device__ uchar x( uchar2 s ) { return s.x; }
        static inline __host__ __device__ uchar y( uchar2 s ) { return s.y; }
        static inline __host__ __device__ uchar z( uchar2 s ) { return 0; }
        static inline __host__ __device__ uchar w( uchar2 s ) { return 0; }
    };

    template<> struct type_traits<uchar3> {
        enum { N = 3 };
        static image_format_t format() { return FMT_UCHAR3; }
        typedef uchar3 value_type;
        typedef uchar scalar_type;
        typedef uchar4 texture_type;

        static inline __host__ __device__ uchar x( uchar3 s ) { return s.x; }
        static inline __host__ __device__ uchar y( uchar3 s ) { return s.y; }
        static inline __host__ __device__ uchar z( uchar3 s ) { return s.z; }
        static inline __host__ __device__ uchar w( uchar3 s ) { return 255; }
    };

    template<> struct type_traits<uchar4> {
        enum { N = 4 };
        static image_format_t format() { return FMT_UCHAR4; }
        typedef uchar4 value_type;
        typedef uchar scalar_type;
        typedef uchar4 texture_type;

        static inline __host__ __device__ uchar x( uchar4 s ) { return s.x; }
        static inline __host__ __device__ uchar y( uchar4 s ) { return s.y; }
        static inline __host__ __device__ uchar z( uchar4 s ) { return s.z; }
        static inline __host__ __device__ uchar w( uchar4 s ) { return s.w; }
    };


    template<> struct type_traits<float> {
        enum { N = 1 };
        static image_format_t format() { return FMT_FLOAT; }
        typedef float value_type;
        typedef float scalar_type;
        typedef float texture_type;

        static inline __host__ __device__ float x( float s ) { return s; }
        static inline __host__ __device__ float y( float s ) { return s; }
        static inline __host__ __device__ float z( float s ) { return s; }
        static inline __host__ __device__ float w( float s ) { return 1; }
    };

    template<> struct type_traits<float2> {
        enum { N = 2 };
        static image_format_t format() { return FMT_FLOAT2; }
        typedef float2 value_type;
        typedef float scalar_type;
        typedef float2 texture_type;

        static inline __host__ __device__ float x( float2 s ) { return s.x; }
        static inline __host__ __device__ float y( float2 s ) { return s.y; }
        static inline __host__ __device__ float z( float2 s ) { return 0; }
        static inline __host__ __device__ float w( float2 s ) { return 0; }
    };

    template<> struct type_traits<float3> {
        enum { N = 3 };
        static image_format_t format() { return FMT_FLOAT3; }
        typedef float3 value_type;
        typedef float scalar_type;
        typedef float4 texture_type;

        static inline __host__ __device__ float x( float3 s ) { return s.x; }
        static inline __host__ __device__ float y( float3 s ) { return s.y; }
        static inline __host__ __device__ float z( float3 s ) { return s.z; }
        static inline __host__ __device__ float w( float3 s ) { return 1; }
    };

    template<> struct type_traits<float4> {
        enum { N = 4 };
        static image_format_t format() { return FMT_FLOAT4; }
        typedef float4 value_type;
        typedef float scalar_type;
        typedef float4 texture_type;

        static inline __host__ __device__ float x( float4 s ) { return s.x; }
        static inline __host__ __device__ float y( float4 s ) { return s.y; }
        static inline __host__ __device__ float z( float4 s ) { return s.z; }
        static inline __host__ __device__ float w( float4 s ) { return s.w; }
    };

}
