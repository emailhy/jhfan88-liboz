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
#include <oz/pyr.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>


namespace {
    template<typename T, int pass> struct PyrDownGauss5x5 : public oz::generator<T> {
        gpu_sampler<T,0> src_;

        PyrDownGauss5x5( const oz::gpu_image& src) : src_(src) {}

        __device__ T operator()( int ix, int iy ) const {
            if (pass == 0) {
                int ux = 2 * ix;
                return 1.0f/16.0f * src_(ux - 2, iy) +
                       4.0f/16.0f * src_(ux - 1, iy) +
                       6.0f/16.0f * src_(ux,     iy) +
                       4.0f/16.0f * src_(ux + 1, iy) +
                       1.0f/16.0f * src_(ux + 2, iy);
            } else {
                int uy = 2 * iy;
                return 1.0f/16.0f * src_(ix, uy - 2) +
                       4.0f/16.0f * src_(ix, uy - 1) +
                       6.0f/16.0f * src_(ix, uy    ) +
                       4.0f/16.0f * src_(ix, uy + 1) +
                       1.0f/16.0f * src_(ix, uy + 2);
            }
        }
    };

    template<typename T> oz::gpu_image pyrdown_gauss5x5T( const oz::gpu_image& src ) {
        int w = (src.w() + 1) / 2;
        int h = (src.h() + 1) / 2;
        oz::gpu_image tmp = oz::generate(w, src.h(), PyrDownGauss5x5<T,0>(src));
        return oz::generate(w, h, PyrDownGauss5x5<T,1>(tmp));
    }


    template<typename T, int pass> struct PyrUpGauss5x5 : public oz::generator<T> {
        gpu_sampler<T,0> src_;

        PyrUpGauss5x5( const oz::gpu_image& src) : src_(src) {}

        __device__ T operator()( int ix, int iy ) const {
            if (pass == 0) {
                int ux = ix / 2;
                T c;
                if (ix & 1) {
                    c = 4.0f/16.0f * src_(ux,     iy) +
                        4.0f/16.0f * src_(ux + 1, iy);
                } else {
                    c = 1.0f/16.0f * src_(ux - 1, iy) +
                        6.0f/16.0f * src_(ux,     iy) +
                        1.0f/16.0f * src_(ux + 1, iy);
                }
                return 2.0f * c;
            } else {
                int uy = iy / 2;
                T c;
                if (iy & 1) {
                    c = 4.0f/16.0f * src_(ix, uy    ) +
                        4.0f/16.0f * src_(ix, uy + 1);
                } else {
                    c = 1.0f/16.0f * src_(ix, uy - 1) +
                        6.0f/16.0f * src_(ix, uy    ) +
                        1.0f/16.0f * src_(ix, uy + 1);
                }
                return 2.0f * c;
            }
        }
    };

    template<typename T> oz::gpu_image pyrup_gauss5x5T( const oz::gpu_image& src, unsigned w, unsigned h ) {
        if (!w) w = 2 * src.w();
        if (!h) h = 2 * src.h();
        oz::gpu_image tmp = oz::generate(w, src.h(), PyrUpGauss5x5<T,0>(src));
        return oz::generate(w, h, PyrUpGauss5x5<T,1>(tmp));
    }
}


oz::gpu_image oz::pyrdown_gauss5x5( const gpu_image& src ) {
    switch (src.format()) {
        case FMT_FLOAT:  return pyrdown_gauss5x5T<float >(src);
        case FMT_FLOAT3: return pyrdown_gauss5x5T<float3>(src);
        case FMT_FLOAT4: return pyrdown_gauss5x5T<float4>(src);
        default:
            OZ_INVALID_FORMAT();
    }
}


oz::gpu_image oz::pyrup_gauss5x5( const gpu_image& src, unsigned w, unsigned h ) {
    switch (src.format()) {
        case FMT_FLOAT:  return pyrup_gauss5x5T<float >(src, w, h);
        case FMT_FLOAT3: return pyrup_gauss5x5T<float3>(src, w, h);
        case FMT_FLOAT4: return pyrup_gauss5x5T<float4>(src, w, h);
        default:
            OZ_INVALID_FORMAT();
    }
}


