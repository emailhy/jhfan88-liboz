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
#include <oz/stgauss3.h>
#include <oz/generate.h>
#include <oz/gpu_sampler2.h>
#include <oz/stintrk2.h>
#include <oz/filter_gauss.h>

namespace oz {

    template<typename T, int order, bool adaptive, class SRC, class ST> struct StGauss3Filter : generator<T> {
        unsigned w_, h_;
        const SRC src_;
        const ST st_;
        float sigma_;
        float step_size_;

        StGauss3Filter( unsigned w, unsigned h, const SRC& src, const ST& st, float sigma, float step_size )
            : w_(w), h_(h), src_(src), st_(st), sigma_(sigma), step_size_(step_size) {}

        inline __device__ T operator()( int ix, int iy ) const {
            float2 p0 = make_float2(ix + 0.5f, iy + 0.5f);
            filter_gauss_1d<T,SRC> f(src_, sigma_);
            st3_int<ST,filter_gauss_1d<T,SRC>,order,adaptive>(p0, st_, f, w_, h_, step_size_);
            return f.result();
        }
    };


    template<typename T, int order>
    gpu_image filterTO( const gpu_image& src, bool src_linear,
                        const gpu_image& st, bool st_linear,
                        float sigma, bool adaptive,
                        float step_size )
    {
        if (adaptive) {
            return generate(src.size(), StGauss3Filter<T, order, true, gpu_sampler<T,0>, gpu_sampler<float3,1> >(
                src.w(), src.h(),
                gpu_sampler<T,0>(src, src_linear? cudaFilterModeLinear : cudaFilterModePoint),
                gpu_sampler<float3,1>(st, st_linear? cudaFilterModeLinear : cudaFilterModePoint),
                sigma, step_size));
        } else {
            return generate(src.size(), StGauss3Filter<T, order, false, gpu_sampler<T,0>, gpu_sampler<float3,1> >(
                src.w(), src.h(),
                gpu_sampler<T,0>(src, src_linear? cudaFilterModeLinear : cudaFilterModePoint),
                gpu_sampler<float3,1>(st, st_linear? cudaFilterModeLinear : cudaFilterModePoint),
                sigma, step_size));
        }
    }


    template<typename T>
    gpu_image filterT( const gpu_image& src, bool src_linear,
                       const gpu_image& st, bool st_linear,
                       float sigma, bool adaptive,
                       int order, float step_size )
    {
        switch (order) {
            case 1: return filterTO<T,1>(src, src_linear, st, st_linear, sigma, adaptive, step_size);
            case 2: return filterTO<T,2>(src, src_linear, st, st_linear, sigma, adaptive, step_size);
            default:
                OZ_X() << "Invalid order!";
        }
    }


    gpu_image stgauss3_filter_( const gpu_image& src, const gpu_image& st,
                               float sigma, bool adaptive, bool src_linear, bool st_linear,
                               int order, float step_size )
    {
        if (sigma <= 0) return src;
        switch (src.format()) {
            case FMT_FLOAT3: return filterT<float3>(src, src_linear, st, st_linear, sigma, adaptive, order, step_size);
            default:
                OZ_INVALID_FORMAT();
        }
    }

}
