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

#include <oz/gpu_image.h>
#include <oz/math_util.h>
#include <oz/type_traits.h>
#include <oz/log.h>

#ifdef __CUDACC__

namespace {

    template<typename T, int unit>
    struct gpu_sampler {
        typedef typename oz::type_traits<T>::texture_type TT;
        typedef typename ::texture<TT,cudaTextureType2D> Texture;
        unsigned w;
        unsigned h;

        __host__ gpu_sampler( const oz::gpu_image& img,
                              cudaTextureFilterMode filter_mode=cudaFilterModePoint,
                              cudaTextureAddressMode address_mode=cudaAddressModeClamp,
                              bool normalized = false )
        {
            OZ_CHECK_FORMAT(img.format(), oz::type_traits<T>::format());
            w = img.w();
            h = img.h();
            texref().filterMode = filter_mode;
            texref().addressMode[0] = address_mode;
            texref().addressMode[1] = address_mode;
            texref().addressMode[2] = address_mode;
            texref().normalized = (int)normalized;
            OZ_CUDA_SAFE_CALL(cudaBindTexture2D(0, texref(), img.ptr<TT>(), img.w(), img.h(), img.pitch()));
        }

        __host__ ~gpu_sampler() {
            texref().filterMode = cudaFilterModePoint;
            OZ_CUDA_SAFE_CALL(cudaUnbindTexture(texref()));
        }

        __device__ Texture texref() const;
        __host__ Texture& texref();

        inline __host__ __device__ T operator()( float x, float y ) const {
            #ifdef __CUDA_ARCH__
            return make_T<T>(tex2D(texref(), x, y));
            #else
            return make_zero<T>();
            #endif
        }

        inline __host__ __device__ T operator()( float2 v ) const {
            return operator()(v.x, v.y);
        }
    };


    template<typename T, int unit>
    struct gpu_resampler : public gpu_sampler<T,unit> {
        typedef typename oz::type_traits<T>::texture_type TT;
        float2 s_;

        __host__ gpu_resampler( const oz::gpu_image& img, float2 s,
                                cudaTextureFilterMode filter_mode=cudaFilterModePoint,
                                cudaTextureAddressMode address_mode=cudaAddressModeClamp,
                                bool normalized = false )
            : gpu_sampler<T,unit>(img, filter_mode, address_mode, normalized), s_(s)
        {}

        inline __host__ __device__ T operator()(float x, float y) const {
            #ifdef __CUDA_ARCH__
            return gpu_sampler<T,unit>::operator()(s_.x * x, s_.x * y);
            #else
            return make_zero<T>();
            #endif
        }

        inline __host__ __device__ T operator()( float2 v ) const {
            return operator()(v.x, v.y);
        }
    };


    template <typename T>
    struct gpu_constant_sampler {
        gpu_constant_sampler(T value) : value_(value) { }

        inline __host__ __device__ T operator()(float ix, float iy) const {
            return value_;
        }

        inline __host__ __device__ T operator()( float2 v ) const {
            return operator()(v.x, v.y);
        }

        T value_;
    };

}

#define OZ_GPU_SAMPLER( ID ) \
    static texture<float, 2> s_tex1##ID; \
    static texture<float2,2> s_tex2##ID; \
    static texture<float4,2> s_tex4##ID; \
    template<> inline __host__ __device__ gpu_sampler<float,ID>::Texture gpu_sampler<float,ID>::texref() const { return s_tex1##ID; } \
    template<> inline __host__ gpu_sampler<float,ID>::Texture& gpu_sampler<float,ID>::texref() { return s_tex1##ID; } \
    template<> inline __host__ __device__ gpu_sampler<float2,ID>::Texture gpu_sampler<float2,ID>::texref() const { return s_tex2##ID; } \
    template<> inline __host__ gpu_sampler<float2,ID>::Texture& gpu_sampler<float2,ID>::texref() { return s_tex2##ID; } \
    template<> inline __host__ __device__ gpu_sampler<float3,ID>::Texture gpu_sampler<float3,ID>::texref() const { return s_tex4##ID; } \
    template<> inline __host__ gpu_sampler<float3,ID>::Texture& gpu_sampler<float3,ID>::texref() { return s_tex4##ID; } \
    template<> inline __host__ __device__ gpu_sampler<float4,ID>::Texture gpu_sampler<float4,ID>::texref() const { return s_tex4##ID; } \
    template<> inline __host__ gpu_sampler<float4,ID>::Texture& gpu_sampler<float4,ID>::texref() { return s_tex4##ID; }

#endif
