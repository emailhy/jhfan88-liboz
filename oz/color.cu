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
#include <oz/color.h>
#include <oz/transform.h>
#include <oz/color_util.h>

namespace oz {
    template<typename T> struct op_srgb2linear : public unary_function<T,T> {
        inline __device__ T operator()( T s ) const {
            return sbgr2linear(s);
        }
    };

    gpu_image srgb2linear( const gpu_image& src) {
        switch (src.format()) {
            case FMT_FLOAT:  return transform(src, op_srgb2linear<float>());
            case FMT_FLOAT3: return transform(src, op_srgb2linear<float3>());
            default:
                OZ_INVALID_FORMAT();
        }
    }


    template<typename T> struct op_linear2srgb : public unary_function<T,T> {
        inline __device__ T operator()( T s ) const {
            return linear2sbgr(s);
        }
    };

    gpu_image linear2srgb( const gpu_image& src) {
        switch (src.format()) {
            case FMT_FLOAT:  return transform(src, op_linear2srgb<float>());;
            case FMT_FLOAT3: return transform(src, op_linear2srgb<float3>());;
            default:
                OZ_INVALID_FORMAT();
        }
    }


    struct op_gray2rgb : public unary_function<float,float3> {
        inline __device__ float3 operator()( float s ) const {
            float c = __saturatef(s);
            return make_float3(c, c, c);
        }
    };

    gpu_image gray2rgb( const gpu_image& src ) {
        return transform(src, op_gray2rgb());
    }


    struct op_rgb2gray {
        inline __device__ float operator()( float3 s ) const {
            return 0.299f * __saturatef(s.z) +
                   0.587f * __saturatef(s.y) +
                   0.114f * __saturatef(s.x);
        }
        inline __device__ float operator()( float4 s ) const {
            return operator()(make_float3(s));
        }
    };

    gpu_image rgb2gray( const gpu_image& src ) {
        switch (src.format()) {
            case FMT_FLOAT3: return transform_unary<float3,float>(src, op_rgb2gray());
            case FMT_FLOAT4: return transform_unary<float4,float>(src, op_rgb2gray());
            default:
                OZ_INVALID_FORMAT();
        }
    }


    struct op_rgb2lab {
        inline __device__ float3 operator()( float3 s ) const {
            return xyz2lab(bgr2xyz(sbgr2linear(s)));
        }
        inline __device__ float4 operator()( float4 s ) const {
            return make_float4(operator()(make_float3(s)), s.w);
        }
    };

    gpu_image rgb2lab( const gpu_image& src) {
        switch (src.format()) {
            case FMT_FLOAT3: return transform_unary<float3,float3>(src, op_rgb2lab());
            case FMT_FLOAT4: return transform_unary<float4,float4>(src, op_rgb2lab());
            default:
                OZ_INVALID_FORMAT();
        }
    }


    struct op_lab2rgb {
        inline __device__ float3 operator()( float3 s ) const {
            return linear2sbgr(xyz2bgr(lab2xyz(s)));
        }
        inline __device__ float4 operator()( float4 s ) const {
            return make_float4(operator()(make_float3(s)), s.w);
        }
    };

    gpu_image lab2rgb( const gpu_image& src) {
        switch (src.format()) {
            case FMT_FLOAT3: return transform_unary<float3,float3>(src, op_lab2rgb());
            case FMT_FLOAT4: return transform_unary<float4,float4>(src, op_lab2rgb());
            default:
                OZ_INVALID_FORMAT();
        }
    }


    struct op_rgb2luv {
        inline __device__ float3 operator()( float3 s ) const {
            return xyz2luv(bgr2xyz(sbgr2linear(s)));
        }
        inline __device__ float4 operator()( float4 s ) const {
            return make_float4(operator()(make_float3(s)), s.w);
        }
    };

    gpu_image rgb2luv( const gpu_image& src) {
        switch (src.format()) {
            case FMT_FLOAT3: return transform_unary<float3,float3>(src, op_rgb2luv());
            case FMT_FLOAT4: return transform_unary<float4,float4>(src, op_rgb2luv());
            default:
                OZ_INVALID_FORMAT();
        }
    }


    struct op_luv2rgb {
        inline __device__ float3 operator()( float3 s ) const {
            return linear2sbgr(xyz2bgr(luv2xyz(s)));
        }
        inline __device__ float4 operator()( float4 s ) const {
            return make_float4(operator()(make_float3(s)), s.w);
        }
    };

    gpu_image luv2rgb( const gpu_image& src) {
        switch (src.format()) {
            case FMT_FLOAT3: return transform_unary<float3,float3>(src, op_luv2rgb());
            case FMT_FLOAT4: return transform_unary<float4,float4>(src, op_luv2rgb());
            default:
                OZ_INVALID_FORMAT();
        }
    }


    struct op_rgb2nvac {
        inline __device__ float operator()( float3 s ) const {
            return luv2nvac(xyz2luv(bgr2xyz(sbgr2linear(s))));
        }
        inline __device__ float operator()( float4 s ) const {
            return operator()(make_float3(s));
        }
    };

    gpu_image rgb2nvac( const gpu_image& src) {
        switch (src.format()) {
            case FMT_FLOAT3: return transform_unary<float3,float>(src, op_rgb2nvac());
            case FMT_FLOAT4: return transform_unary<float4,float>(src, op_rgb2nvac());
            default:
                OZ_INVALID_FORMAT();
        }
    }


    struct op_swap_rgb {
        inline __device__ uchar3 operator()( uchar3 s ) const {
            return make_uchar3(s.z, s.y, s.x);
        }
        inline __device__ uchar4 operator()( uchar4 s ) const {
            return make_uchar4(s.z, s.y, s.x, s.w);
        }
        inline __device__ float3 operator()( float3 s ) const {
            return make_float3(s.z, s.y, s.x);
        }
        inline __device__ float4 operator()( float4 s ) const {
            return make_float4(s.z, s.y, s.x, s.w);
        }
    };

    gpu_image swap_rgb( const gpu_image& src) {
        switch (src.format()) {
            case FMT_UCHAR3: return transform_unary<uchar3,uchar3>(src, op_swap_rgb());
            case FMT_UCHAR4: return transform_unary<uchar4,uchar4>(src, op_swap_rgb());
            case FMT_FLOAT3: return transform_unary<float3,float3>(src, op_swap_rgb());
            case FMT_FLOAT4: return transform_unary<float4,float4>(src, op_swap_rgb());
            default:
                OZ_INVALID_FORMAT();
        }
    }
}
