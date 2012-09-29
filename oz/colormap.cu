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
#include <oz/colormap.h>
#include <oz/transform.h>
#include <oz/colormap_util.h>

namespace oz {

    struct ColormapJet : public oz::unary_function<float,float3> {
        float scale_;

        ColormapJet( float scale) : scale_(scale) {}

        inline __device__ float3 operator()( float c ) const {
            return colormap_jet(c * scale_);
        }
    };

    gpu_image colormap_jet( const gpu_image& src, float scale ) {
        return transform(src, ColormapJet(scale));
    }


    struct ColormapJet2 : public oz::unary_function<float,float3> {
        float scale_;

        ColormapJet2( float scale) : scale_(scale) {}

        inline __device__ float3 operator()( float c ) const {
            return colormap_jet2(c * scale_);
        }
    };

    gpu_image colormap_jet2( const gpu_image& src, float scale ) {
        return transform(src, ColormapJet2(scale));
    }


    template<typename T> struct ColormapDiff : public oz::binary_function<T,T,float3> {
        float scale_;

        ColormapDiff( float scale) : scale_(scale) {}

        inline __device__ float3 operator()( T a, T b ) const {
            float e = length(a - b);
            return colormap_jet( clamp(e, 0.0f, scale_) / scale_ );
        }
    };

    gpu_image colormap_diff( const gpu_image& src0, const gpu_image& src1, float scale ) {
        switch (src0.format()) {
            case FMT_FLOAT:  return transform(src0, src1, ColormapDiff<float >(scale));
            case FMT_FLOAT2: return transform(src0, src1, ColormapDiff<float2>(scale));
            case FMT_FLOAT3: return transform(src0, src1, ColormapDiff<float3>(scale));
            default:
                OZ_INVALID_FORMAT();
        }
    }

}


