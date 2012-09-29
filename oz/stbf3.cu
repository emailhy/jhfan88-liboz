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
#include <oz/stbf3.h>
#include <oz/st.h>
#include <oz/st_util.h>
#include <oz/stintrk2.h>
#include <oz/filter_bf.h>
#include <oz/generate.h>
#include <oz/gpu_sampler2.h>


namespace oz {

    template<typename T, int order, bool ustep, bool src_linear, class SRC, class ST> struct StBf3Filter : generator<T> {
        unsigned w_, h_;
        const SRC src_;
        const ST st_;
        float sigma_r_, sigma_d_, precision_;
        float step_size_;

        StBf3Filter( unsigned w, unsigned h, const SRC& src, const ST& st,
                     float sigma_d, float sigma_r, float precision, float step_size )
            : w_(w), h_(h), src_(src), st_(st), sigma_d_(sigma_d),
              sigma_r_(sigma_r), precision_(precision), step_size_(step_size) {}

        inline __device__ T operator()( int ix, int iy ) const {
            float2 p0 = make_float2(ix + 0.5f, iy + 0.5f);
            if (ustep) {
                filter_bf_trapez<T,SRC,src_linear> f(src_, src_(p0.x, p0.y), sigma_d_, sigma_r_, precision_);
                st3_int_ustep<ST,filter_bf_trapez<T,SRC,src_linear>,order>(p0, st_, f, w_, h_, step_size_);
                return f.result();
            } else {
                filter_bf<T,SRC> f(src_, src_(p0.x, p0.y), sigma_d_, sigma_r_, precision_);
                st3_int<ST,filter_bf<T,SRC>,order,false>(p0, st_, f, w_, h_, step_size_);
                return f.result();
            }
        }
    };


    template<typename T, int order>
    gpu_image filterTO( const oz::gpu_image& src, bool src_linear,
                        const oz::gpu_image& st, bool st_linear,
                        float sigma_d, float sigma_r, float precision,
                        bool ustep,
                        float step_size )
    {
        if (ustep) {
            if (src_linear) {
                return generate(src.size(), StBf3Filter<T, order, true, true, gpu_sampler<T,0>, gpu_sampler<float3,1> >(
                    src.w(), src.h(),
                    gpu_sampler<T,0>(src, cudaFilterModeLinear),
                    gpu_sampler<float3,1>(st, st_linear? cudaFilterModeLinear : cudaFilterModePoint),
                    sigma_d, sigma_r, precision, step_size));
            } else {
                return generate(src.size(), StBf3Filter<T, order, true, false, gpu_sampler<T,0>, gpu_sampler<float3,1> >(
                    src.w(), src.h(),
                    gpu_sampler<T,0>(src, cudaFilterModePoint),
                    gpu_sampler<float3,1>(st, st_linear? cudaFilterModeLinear : cudaFilterModePoint),
                    sigma_d, sigma_r, precision, step_size));
            }
        } else {
            return generate(src.size(), StBf3Filter<T, order, false, false, gpu_sampler<T,0>, gpu_sampler<float3,1> >(
                src.w(), src.h(),
                gpu_sampler<T,0>(src, src_linear? cudaFilterModeLinear : cudaFilterModePoint),
                gpu_sampler<float3,1>(st, st_linear? cudaFilterModeLinear : cudaFilterModePoint),
                sigma_d, sigma_r, precision, step_size));
        }
    }


    template<typename T>
    gpu_image filterT( const oz::gpu_image& src, bool src_linear,
                       const oz::gpu_image& st, bool st_linear,
                       float sigma_d, float sigma_r, float precision,
                       bool ustep, int order, float step_size )
    {
        switch (order) {
            case 1: return filterTO<T,1>(src, src_linear, st, st_linear, sigma_d, sigma_r, precision, ustep, step_size);
            case 2: return filterTO<T,2>(src, src_linear, st, st_linear, sigma_d, sigma_r, precision, ustep, step_size);
            default:
                OZ_X() << "Invalid order!";
        }
    }


    gpu_image stbf3_filter_( const gpu_image& src, const gpu_image& st,
                             float sigma_d, float sigma_r, float precision,
                             bool src_linear, bool st_linear, bool ustep, int order, float step_size )
    {
        if (sigma_d <= 0) return src;
        if (src.size() != st.size()) OZ_X() << "Sizes must match!";
        switch (src.format()) {
            case FMT_FLOAT:  return filterT<float >(src, src_linear, st, st_linear, sigma_d, sigma_r, precision, ustep, order, step_size);
            case FMT_FLOAT3: return filterT<float3>(src, src_linear, st, st_linear, sigma_d, sigma_r, precision, ustep, order, step_size);
            default:
                OZ_INVALID_FORMAT();
        }
    }

}
