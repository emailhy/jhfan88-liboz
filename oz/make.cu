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
#include <oz/make.h>
#include <oz/transform.h>


namespace {
    inline __device__ float2 make(float  a, float  b) { return make_float2(a, b); }
    inline __device__ float3 make(float  a, float2 b) { return make_float3(a, b.x, b.y); }
    inline __device__ float4 make(float  a, float3 b) { return make_float4(a, b.x, b.y, b.z); }
    inline __device__ float3 make(float2 a, float  b) { return make_float3(a, b); }
    inline __device__ float4 make(float2 a, float2 b) { return make_float4(a.x, a.y, b.x, b.y); }
    inline __device__ float4 make(float3 a, float  b) { return make_float4(a, b); }

    template<typename Arg1, typename Arg2, typename Result>
    struct Make2 : public oz::binary_function<Arg1,Arg2,Result> {
        inline __device__ Result operator()( Arg1 a, Arg2 b ) const {
            return make(a, b);
        }
    };

    template<typename Arg, typename Result>
    struct Make2c : public oz::unary_function<Arg,Result>{
        float c_;
        Make2c( float c ) : c_(c) {}
        inline __device__ Result operator()( Arg a ) const {
            return make(a, c_);
        }
    };

    struct Make3 : public oz::ternary_function<float,float,float,float3> {
        inline __device__ float3 operator()( float a, float b, float c ) const {
            return make_float3(a, b, c);
        }
    };

    struct Make4 : public oz::quaternary_function<float,float,float,float,float4> {
        inline __device__ float4 operator()( float a, float b, float c, float d ) const {
            return make_float4(a, b, c, d);
        }
    };
}


oz::gpu_image oz::make( const gpu_image& src0, const gpu_image& src1 ) {
    if (src0.size() != src1.size()) OZ_INVALID_SIZE();
    switch (src0.format()) {
        case FMT_FLOAT:
            switch (src1.format()) {
                case FMT_FLOAT:  return transform(src0, src1, Make2<float,float, float2>());
                case FMT_FLOAT2: return transform(src0, src1, Make2<float,float2,float3>());
                case FMT_FLOAT3: return transform(src0, src1, Make2<float,float3,float4>());
            }
            break;
        case FMT_FLOAT2:
            switch (src1.format()) {
                case FMT_FLOAT:  return transform(src0, src1, Make2<float2,float, float3>());
                case FMT_FLOAT2: return transform(src0, src1, Make2<float2,float2,float4>());
            }
            break;
        case FMT_FLOAT3:
            switch (src1.format()) {
                case FMT_FLOAT:  return transform(src0, src1, Make2<float3,float,float4>());
            }
            break;
    }
    OZ_INVALID_FORMAT();
}


oz::gpu_image oz::make( const gpu_image& src0, float c1 ) {
    switch (src0.format()) {
        case FMT_FLOAT:  return transform(src0, Make2c<float, float2>(c1));
        case FMT_FLOAT2: return transform(src0, Make2c<float2,float3>(c1));
        case FMT_FLOAT3: return transform(src0, Make2c<float3,float4>(c1));
        default:
            OZ_INVALID_FORMAT();
    }
}


oz::gpu_image oz::make( const gpu_image& src0, const gpu_image& src1, const gpu_image& src2 ) {
    return transform(src0, src1, src2, Make3());
}


oz::gpu_image oz::make( const gpu_image& src0, const gpu_image& src1, const gpu_image& src2, const gpu_image& src3 ) {
    return transform(src0, src1, src2, src3, Make4());
}
