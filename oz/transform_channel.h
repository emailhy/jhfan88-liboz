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

#include <oz/transform.h>

namespace oz {

    template<typename F> struct ChannelUnary {
        const F f_;
        ChannelUnary(const F &f) : f_(f) {}
        inline __device__ float operator()( float s ) const {
            return f_(s);
        }
        inline __device__ float2 operator()( float2 s ) const {
            return make_float2(f_(s.x), f_(s.y));
        }
        inline __device__ float3 operator()( float3 s ) const {
            return make_float3(f_(s.x), f_(s.y), f_(s.z));
        }
        inline __device__ float4 operator()( float4 s ) const {
            return make_float4(f_(s.x), f_(s.y), f_(s.z), f_(s.w));
        }
    };

    template<typename F> struct ChannelBinary {
        const F f_;
        ChannelBinary( const F &f ) : f_(f) {}
        inline __device__ float operator()( float a, float b ) const {
            return f_(a, b);
        }
        inline __device__ float2 operator()( float2 a, float2 b ) const {
            return make_float2(f_(a.x, b.x), f_(a.y, b.y));
        }
        inline __device__ float3 operator()( float3 a, float3 b ) const {
            return make_float3(f_(a.x, b.x), f_(a.y, b.y), f_(a.z, b.z));
        }
        inline __device__ float4 operator()( float4 a, float4 b ) const {
            return make_float4(f_(a.x, b.x), f_(a.y, b.y), f_(a.z, b.z), f_(a.w, b.w));
        }
    };

    template<typename T, typename F>
    gpu_image transform_channel( const gpu_image& src, const F& f) {
        return transform_unary<T,T>(src, ChannelUnary<F>(f));
    }

    template<typename T, typename F>
    gpu_image transform_channel( const gpu_image& src0, const gpu_image& src1, const F& f) {
        return transform_binary<T,T,T>(src0, src1, ChannelBinary<F>(f));
    }

    template<typename F>
    gpu_image transform_channel_f( const gpu_image& src, const F& f) {
        switch (src.format()) {
            case FMT_FLOAT:  return transform_channel<float >(src, f);
            case FMT_FLOAT2: return transform_channel<float2>(src, f);
            case FMT_FLOAT3: return transform_channel<float3>(src, f);
            case FMT_FLOAT4: return transform_channel<float4>(src, f);
            default:
                OZ_INVALID_FORMAT();
        }
    }

    template<typename F>
    gpu_image transform_channel_f( const gpu_image& src0, const gpu_image& src1, const F& f) {
        switch (src0.format()) {
            case FMT_FLOAT:  return transform_channel<float >(src0, src1, f);
            case FMT_FLOAT2: return transform_channel<float2>(src0, src1, f);
            case FMT_FLOAT3: return transform_channel<float3>(src0, src1, f);
            case FMT_FLOAT4: return transform_channel<float4>(src0, src1, f);
            default:
                OZ_INVALID_FORMAT();
        }
    }

}
