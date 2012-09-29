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
#include <oz/resize.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>
#include <oz/tex2d_util.h>


namespace oz {
    template <typename T, resize_mode_t mode> struct Tex2D;

    template <typename T> struct Tex2D<T,RESIZE_NEAREST> : public generator<T> {
        gpu_sampler<T,0> src_;
        float2 s_;
        Tex2D( const gpu_image& src, float2 s ) : src_(src, cudaFilterModePoint), s_(s) {}

        inline __device__ T operator()( int ix, int iy ) const {
            return src_( s_.x * (ix + 0.5f), s_.y * (iy + 0.5f) );
        }
    };


    template <typename T> struct Tex2D<T,RESIZE_FAST_BILINEAR> : public generator<T> {
        gpu_sampler<T,0> src_;
        float2 s_;
        Tex2D( const gpu_image& src, float2 s ) : src_(src, cudaFilterModeLinear), s_(s) {}

        inline __device__ T operator()( int ix, int iy ) const {
            return src_( s_.x * (ix + 0.5f), s_.y * (iy + 0.5f) );
        }
    };


    template <typename T> struct Tex2D<T,RESIZE_BILINEAR> : public generator<T> {
        gpu_sampler<T,0> src_;
        float2 s_;
        Tex2D( const gpu_image& src, float2 s ) : src_(src, cudaFilterModePoint), s_(s) {}

        inline __device__ T operator()( int ix, int iy ) const {
            return make_T<T>(tex2DBilinear(src_.texref(), s_.x * (ix + 0.5f), s_.y * (iy + 0.5f)));
        }
    };


    template <typename T> struct Tex2D<T,RESIZE_FAST_BICUBIC> : public generator<T> {
        gpu_sampler<T,0> src_;
        float2 s_;
        Tex2D( const gpu_image& src, float2 s ) : src_(src, cudaFilterModeLinear), s_(s) {}

        inline __device__ T operator()( int ix, int iy ) const {
            return make_T<T>(tex2DFastBicubic(src_.texref(), s_.x * (ix + 0.5f), s_.y * (iy + 0.5f)));
        }
    };


    template <typename T> struct Tex2D<T,RESIZE_BICUBIC> : public generator<T> {
        gpu_sampler<T,0> src_;
        float2 s_;
        Tex2D( const gpu_image& src, float2 s ) : src_(src, cudaFilterModePoint), s_(s) {}

        inline __device__ T operator()( int ix, int iy ) const {
            return make_T<T>(tex2DBicubic(src_.texref(), s_.x * (ix + 0.5f), s_.y * (iy + 0.5f)));
        }
    };


    template <typename T> struct Tex2D<T,RESIZE_CATROM> : public generator<T> {
        gpu_sampler<T,0> src_;
        float2 s_;
        Tex2D( const gpu_image& src, float2 s ) : src_(src, cudaFilterModePoint), s_(s) {}

        inline __device__ T operator()( int ix, int iy ) const {
            return make_T<T>(tex2DCatRom(src_.texref(), s_.x * (ix + 0.5f), s_.y * (iy + 0.5f)));
        }
    };


    template <typename T>
    gpu_image resizeT( const gpu_image& src, unsigned w, unsigned h, resize_mode_t mode ) {
        float2 s = make_float2((float)src.w() / w, (float)src.h() / h);
        switch (mode) {
            case RESIZE_NEAREST:       return generate(w, h, Tex2D<T,RESIZE_NEAREST>(src, s));
            case RESIZE_FAST_BILINEAR: return generate(w, h, Tex2D<T,RESIZE_FAST_BILINEAR>(src, s));
            case RESIZE_BILINEAR:      return generate(w, h, Tex2D<T,RESIZE_BILINEAR>(src, s));
            case RESIZE_FAST_BICUBIC:  return generate(w, h, Tex2D<T,RESIZE_FAST_BICUBIC>(src, s));
            case RESIZE_BICUBIC:       return generate(w, h, Tex2D<T,RESIZE_BICUBIC>(src, s));
            case RESIZE_CATROM:        return generate(w, h, Tex2D<T,RESIZE_CATROM>(src, s));
        }
        OZ_INVALID_FORMAT();
    }
}


oz::gpu_image oz::resize( const gpu_image& src, unsigned w, unsigned h, resize_mode_t mode ) {
    switch (src.format()) {
        case FMT_FLOAT:   return resizeT<float >(src, w, h, mode);
        case FMT_FLOAT2:  return resizeT<float2>(src, w, h, mode);
        case FMT_FLOAT3:  return resizeT<float3>(src, w, h, mode);
        case FMT_FLOAT4:  return resizeT<float4>(src, w, h, mode);
        default:
            OZ_INVALID_FORMAT();
    }
}


namespace oz {
    template <typename T> struct ResizeX2 : public generator<T> {
        gpu_sampler<T,0> src_;

        ResizeX2( const gpu_image& src) : src_(src) {}

        inline __device__ T operator()( int ix, int iy ) const {
            return src_(ix/2, iy/2);
        }
    };

    template <typename T> struct ResizeHalf : public generator<T> {
        gpu_sampler<T,0> src_;

        ResizeHalf( const gpu_image& src) : src_(src) {}

        inline __device__ T operator()( int ix, int iy ) const {
            T c0 = src_(2*ix,   2*iy);
            T c1 = src_(2*ix+1, 2*iy);
            T c2 = src_(2*ix,   2*iy+1);
            T c3 = src_(2*ix+1, 2*iy+1);
            return 0.25f * (c0 + c1 + c2 + c3);
        }
    };
}


oz::gpu_image oz::resize_x2( const gpu_image& src ) {
    switch (src.format()) {
        case FMT_FLOAT:  return generate(2*src.w(), 2*src.h(), ResizeX2<float >(src));
        case FMT_FLOAT3: return generate(2*src.w(), 2*src.h(), ResizeX2<float3>(src));
        default:
            OZ_INVALID_FORMAT();
    }
}


oz::gpu_image oz::resize_half( const gpu_image& src ) {
    switch (src.format()) {
        case FMT_FLOAT:  return generate(src.w()/2, src.h()/2, ResizeHalf<float >(src));
        case FMT_FLOAT3: return generate(src.w()/2, src.h()/2, ResizeHalf<float3>(src));
        default:
            OZ_INVALID_FORMAT();
    }
}
