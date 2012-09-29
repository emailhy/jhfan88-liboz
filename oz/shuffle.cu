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
#include <oz/shuffle.h>
#include <oz/transform.h>


namespace {

    __forceinline__ __device__ float at( float s, int index ) {
        return (index==0)? s : 0;
    }

    __forceinline__ __device__ float at( float2 s, int index ) {
        return (index==0)? s.x :
               (index==1)? s.y : 0;
    }

    __forceinline__ __device__ float at( float3 s, int index ) {
        return (index==0)? s.x :
               (index==1)? s.y :
               (index==2)? s.z : 0;
    }

    __forceinline__ __device__ float at( float4 s, int index ) {
        return (index==0)? s.x :
            (index==1)? s.y :
            (index==2)? s.z :
            (index==3)? s.w : 0;
    }

    template<typename T> struct op_shuffle1 : public oz::unary_function<T,float> {
        int index_;

        op_shuffle1(int index) : index_(index) {}

        inline __device__ float operator()( T s ) const {
            return at(s, index_);
        }
    };

    template<typename T> struct op_shuffle2 : public oz::unary_function<T,float2>{
        int2 index_;

        op_shuffle2(int2 index) : index_(index) {}

        inline __device__ float2 operator()( T s ) const {
            return make_float2( at(s, index_.x), at(s, index_.y) );
        }
    };

    template<typename T> struct op_shuffle3 : public oz::unary_function<T,float3> {
        int3 index_;

        op_shuffle3(int3 index) : index_(index) {}

        inline __device__ float3 operator()( T s ) const {
            return make_float3( at(s, index_.x), at(s, index_.y), at(s, index_.z) );
        }
    };

    template<typename T> struct op_shuffle4 : public oz::unary_function<T,float4> {
        int4 index_;

        op_shuffle4(int4 index) : index_(index) {}

        inline __device__ float4 operator()( T s ) const {
            return make_float4( at(s, index_.x), at(s, index_.y),
                                at(s, index_.z), at(s, index_.w) );
        }
    };

}


oz::gpu_image oz::shuffle( const gpu_image& src, int x ) {
    switch (src.format()) {
        case oz::FMT_FLOAT2: return transform(src, op_shuffle1<float2>(x));
        case oz::FMT_FLOAT3: return transform(src, op_shuffle1<float3>(x));
        case oz::FMT_FLOAT4: return transform(src, op_shuffle1<float4>(x));
        default:
            OZ_INVALID_FORMAT();
    }
}


oz::gpu_image oz::shuffle( const gpu_image& src, int x, int y ) {
    int2 index = make_int2(x,y);
    switch (src.format()) {
        case oz::FMT_FLOAT:  return transform(src, op_shuffle2<float>(index));
        case oz::FMT_FLOAT2: return transform(src, op_shuffle2<float2>(index));
        case oz::FMT_FLOAT3: return transform(src, op_shuffle2<float3>(index));
        case oz::FMT_FLOAT4: return transform(src, op_shuffle2<float4>(index));
        default:
            OZ_INVALID_FORMAT();
    }
}


oz::gpu_image oz::shuffle( const gpu_image& src, int x, int y, int z ) {
    int3 index = make_int3(x,y,z);
    switch (src.format()) {
        case oz::FMT_FLOAT:  return transform(src, op_shuffle3<float>(index));
        case oz::FMT_FLOAT2: return transform(src, op_shuffle3<float2>(index));
        case oz::FMT_FLOAT3: return transform(src, op_shuffle3<float3>(index));
        case oz::FMT_FLOAT4: return transform(src, op_shuffle3<float4>(index));
        default:
            OZ_INVALID_FORMAT();
    }
}


oz::gpu_image oz::shuffle( const gpu_image& src, int x, int y, int z, int w ) {
    int4 index = make_int4(x,y,z,w);
    switch (src.format()) {
        case oz::FMT_FLOAT:  return transform(src, op_shuffle4<float>(index));
        case oz::FMT_FLOAT2: return transform(src, op_shuffle4<float2>(index));
        case oz::FMT_FLOAT3: return transform(src, op_shuffle4<float3>(index));
        case oz::FMT_FLOAT4: return transform(src, op_shuffle4<float4>(index));
        default:
            OZ_INVALID_FORMAT();
    }
}
