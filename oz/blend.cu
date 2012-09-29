//
// by Jan Eric Kyprianidis and Daniel Müller
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
#include <oz/blend.h>
#include <oz/transform.h>


namespace oz{
    template<blend_mode_t M> inline __device__ float blend(float b, float s);

    template<> inline __device__  float blend<BLEND_NORMAL>(float b, float s) {
        return s;
    }

    template<> inline __device__ float blend<BLEND_MULTIPLY>(float b, float s) {
        return b * s;
    }

    template<> inline __device__ float blend<BLEND_SCREEN>(float b, float s) {
        return b + s - (b * s);
    }

    template<> inline __device__ float blend<BLEND_HARD_LIGHT>(float b, float s) {
        return (s <= 0.5f)? blend<BLEND_MULTIPLY>(b, 2 * s)
                          : blend<BLEND_SCREEN>(b, 2 * s - 1);
    }

    inline __device__ float D(float x) {
        return (x <= 0.25f)? ((16 * x - 12) * x + 4) * x
                           : sqrtf(x);
    }

    template<> inline __device__ float blend<BLEND_SOFT_LIGHT>(float b, float s) {
        return (s <= 0.5f)? b - (1 - 2 * s) * b * (1 - b)
                          : b + (2 * s - 1) * (D(b) - b);
    }

    template<> inline __device__ float blend<BLEND_OVERLAY>(float b, float s) {
        return blend<BLEND_HARD_LIGHT>(s, b);
    }

    template<> inline __device__  float blend<BLEND_LINEAR_BURN>(float b, float s) {
        return b + s - 1;
    }

    template<> inline __device__  float blend<BLEND_DIFFERENCE>(float b, float s) {
        return fabs(b - s);
    }

    template<> inline __device__  float blend<BLEND_LINEAR_DODGE>(float b, float s) {
        return b + s;
    }

    template<blend_mode_t M> struct op_blend {
        __device__ float4 operator()( float4 back, float4 src ) const {
            const float3 b = make_float3(back);
            const float3 s = make_float3(src);

            const float  ab = back.w;
            const float  as = src.w;
            const float  ar = ab + as - (ab * as);

            const float3 bs = make_float3(
                blend<M>(b.x, s.x),
                blend<M>(b.y, s.y),
                blend<M>(b.z, s.z) );

            float3 r = (1 - as / ar) * b + (as / ar) * ((1 - ab) * s + ab * bs);
            return make_float4(__saturatef(r.x), __saturatef(r.y),
                               __saturatef(r.z), __saturatef(ar));
        }

        __device__ float3 operator()( float3 b, float3 s ) const {
            return make_float3(__saturatef(blend<M>(b.x, s.x)),
                               __saturatef(blend<M>(b.y, s.y)),
                               __saturatef(blend<M>(b.z, s.z)));
        }

        __device__ float operator()( float b, float s ) const {
            return __saturatef(blend<M>(b, s));
        }
    };


    template<blend_mode_t M, typename T> struct op_blend_intensity;

    template<blend_mode_t M> struct op_blend_intensity<M,float4> {
        float4 color_;

        op_blend_intensity( float4 color )
            : color_(color) {}

        __device__ float4 operator()( float4 back, float src ) const {
            const float3 b = make_float3(back);
            const float3 s = make_float3(color_) * src;

            const float  ab = back.w;
            const float  as = color_.w;
            const float  ar = ab + as - (ab * as);

            const float3 bs = make_float3(
                blend<M>(b.x, s.x),
                blend<M>(b.y, s.y),
                blend<M>(b.z, s.z) );

            float3 r = (1 - as / ar) * b + (as / ar) * ((1 - ab) * s + ab * bs);
            return make_float4(__saturatef(r.x), __saturatef(r.y),
                               __saturatef(r.z), __saturatef(ar));
        }
    };

    template<blend_mode_t M> struct op_blend_intensity<M,float3> {
        float3 color_;

        op_blend_intensity( float3 color )
            : color_(color) {}

        __device__ float3 operator()( float3 b, float src ) const {
            const float3 s = color_ * src;
            return make_float3(__saturatef(blend<M>(b.x, s.x)),
                               __saturatef(blend<M>(b.y, s.y)),
                               __saturatef(blend<M>(b.z, s.z)) );
        }
    };


    template<typename T>
    static gpu_image blendT( const gpu_image& back, const gpu_image& src, blend_mode_t mode ) {
        switch(mode) {
            case BLEND_NORMAL:       return transform_binary<T,T,T>(back, src, op_blend<BLEND_NORMAL>());
            case BLEND_MULTIPLY:     return transform_binary<T,T,T>(back, src, op_blend<BLEND_MULTIPLY>());
            case BLEND_LINEAR_BURN:  return transform_binary<T,T,T>(back, src, op_blend<BLEND_LINEAR_BURN>());
            case BLEND_SCREEN:       return transform_binary<T,T,T>(back, src, op_blend<BLEND_SCREEN>());
            case BLEND_HARD_LIGHT:   return transform_binary<T,T,T>(back, src, op_blend<BLEND_HARD_LIGHT>());
            case BLEND_SOFT_LIGHT:   return transform_binary<T,T,T>(back, src, op_blend<BLEND_SOFT_LIGHT>());
            case BLEND_OVERLAY:      return transform_binary<T,T,T>(back, src, op_blend<BLEND_OVERLAY>());
            case BLEND_DIFFERENCE:   return transform_binary<T,T,T>(back, src, op_blend<BLEND_DIFFERENCE>());
            case BLEND_LINEAR_DODGE: return transform_binary<T,T,T>(back, src, op_blend<BLEND_LINEAR_DODGE>());
            default:
                OZ_X() << "Unsupported blend mode!";
        }
    }


    gpu_image blend( const gpu_image& back, const gpu_image& src, blend_mode_t mode ) {
        if (back.size() != src.size()) OZ_INVALID_SIZE();
        if (back.format() != src.format()) OZ_INVALID_SIZE();
        switch (src.format()) {
            case FMT_FLOAT:  return blendT<float >(back, src, mode);
            case FMT_FLOAT3: return blendT<float3>(back, src, mode);
            case FMT_FLOAT4: return blendT<float4>(back, src, mode);
            default:
                OZ_INVALID_FORMAT();
        }
    }


    template<typename T>
    static gpu_image blend_intensityT( const gpu_image& back, const gpu_image& src,
                                           blend_mode_t mode, T color) {
        switch(mode) {
            case BLEND_NORMAL:       return transform_binary<T,float,T>(back, src, op_blend_intensity<BLEND_NORMAL,T>(color));
            case BLEND_MULTIPLY:     return transform_binary<T,float,T>(back, src, op_blend_intensity<BLEND_MULTIPLY,T>(color));
            case BLEND_LINEAR_BURN:  return transform_binary<T,float,T>(back, src, op_blend_intensity<BLEND_LINEAR_BURN,T>(color));
            case BLEND_SCREEN:       return transform_binary<T,float,T>(back, src, op_blend_intensity<BLEND_SCREEN,T>(color));
            case BLEND_HARD_LIGHT:   return transform_binary<T,float,T>(back, src, op_blend_intensity<BLEND_HARD_LIGHT,T>(color));
            case BLEND_SOFT_LIGHT:   return transform_binary<T,float,T>(back, src, op_blend_intensity<BLEND_SOFT_LIGHT,T>(color));
            case BLEND_OVERLAY:      return transform_binary<T,float,T>(back, src, op_blend_intensity<BLEND_OVERLAY,T>(color));
            case BLEND_DIFFERENCE:   return transform_binary<T,float,T>(back, src, op_blend_intensity<BLEND_DIFFERENCE,T>(color));
            case BLEND_LINEAR_DODGE: return transform_binary<T,float,T>(back, src, op_blend_intensity<BLEND_LINEAR_DODGE,T>(color));
            default:
                OZ_X() << "Unsupported blend mode!";
        }
    }


    gpu_image blend_intensity( const gpu_image& back, const gpu_image& src,
                                       blend_mode_t mode, float4 color )
    {
        if (back.size() != src.size()) OZ_INVALID_SIZE();
        switch (back.format()) {
            case FMT_FLOAT3: return blend_intensityT<float3>(back, src, mode, make_float3(color));
            case FMT_FLOAT4: return blend_intensityT<float4>(back, src, mode, color);
            default:
                OZ_INVALID_FORMAT();
        }
    }
}