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
#include <oz/stbf2.h>
#include <oz/st.h>
#include <oz/st_util.h>
#include <oz/stintrk.h>
#include <oz/filter_bf.h>
#include <oz/generate.h>
#include <oz/gpu_sampler2.h>

namespace oz {

    template<typename T, int order, class SRC, class ST> struct StBf2Filter : generator<T> {
        unsigned w_, h_;
        const SRC src_;
        const ST st_;
        float sigma_r_, sigma_d_, precision_;
        float cos_max_;
        bool adaptive_;
        float step_size_;

        StBf2Filter( unsigned w, unsigned h, const SRC& src, const ST& st,
                     float sigma_d, float sigma_r, float precision,
                     float cos_max, bool adaptive, float step_size )
            : w_(w), h_(h), src_(src), st_(st),
              sigma_d_(sigma_d), sigma_r_(sigma_r), precision_(precision),
              cos_max_(cos_max), adaptive_(adaptive), step_size_(step_size) {}

        inline __device__ T operator()( int ix, int iy ) const {
            float2 p0 = make_float2(ix + 0.5f, iy + 0.5f);
            float sigma_d = sigma_d_;
            if (adaptive_) {
                float A = st2A(st_(p0.x, p0.y));
                sigma_d *= 0.25f * (1 + A)*(1 + A);
            }
            filter_bf<T,SRC> f(src_, src_(p0.x, p0.y), sigma_d, sigma_r_, precision_);
            if (order == 1) st_integrate_euler(p0, st_, f, cos_max_, w_, h_, step_size_);
            if (order == 2) st_integrate_rk2(p0, st_, f, cos_max_, w_, h_, step_size_);
            if (order == 4) st_integrate_rk4(p0, st_, f, cos_max_, w_, h_, step_size_);
            return f.result();
        }
    };


    template<typename T, int order>
    gpu_image filterTO( const gpu_image& src, bool src_linear,
                        const gpu_image& st, bool st_linear,
                        float sigma_d, float sigma_r, float precision,
                        float max_angle, bool adaptive, float step_size )
    {
        float cos_max = cosf(radians(max_angle));
        if (src.size() == st.size()) {
            return generate(src.size(), StBf2Filter<T, order, gpu_sampler<T,0>, gpu_sampler<float3,1> >(
                src.w(), src.h(),
                gpu_sampler<T,0>(src, src_linear? cudaFilterModeLinear : cudaFilterModePoint),
                gpu_sampler<float3,1>(st, st_linear? cudaFilterModeLinear : cudaFilterModePoint),
                sigma_d, sigma_r, precision, cos_max, adaptive, step_size));
        } else {
            float2 s = make_float2((float)st.w() / src.w(), (float)st.h() / src.h());
            return generate(src.size(), StBf2Filter<T, order, gpu_sampler<T,0>, gpu_resampler<float3,1> >(
                src.w(), src.h(),
                gpu_sampler<T,0>(src, src_linear? cudaFilterModeLinear : cudaFilterModePoint),
                gpu_resampler<float3,1>(st, s, st_linear? cudaFilterModeLinear : cudaFilterModePoint),
                sigma_d, sigma_r, precision, cos_max, adaptive, step_size));
        }
    }


    template<typename T>
    gpu_image filterT( const gpu_image& src, bool src_linear,
                       const gpu_image& st, bool st_linear,
                       float sigma_d, float sigma_r, float precision,
                       float max_angle, bool adaptive,
                       int order, float step_size )
    {
        switch (order) {
            case 1: return filterTO<T,1>(src, src_linear, st, st_linear, sigma_d, sigma_r, precision, max_angle, adaptive, step_size);
            case 2: return filterTO<T,2>(src, src_linear, st, st_linear, sigma_d, sigma_r, precision, max_angle, adaptive, step_size);
            case 4: return filterTO<T,4>(src, src_linear, st, st_linear, sigma_d, sigma_r, precision, max_angle, adaptive, step_size);
            default:
                OZ_X() << "Invalid order!";
        }
    }


    gpu_image stbf2_filter( const gpu_image& src, const gpu_image& st,
                            float sigma_d, float sigma_r, float precision,
                            float max_angle, bool adaptive,
                            bool src_linear, bool st_linear, int order, float step_size )
    {
        if (sigma_d <= 0) return src;
        switch (src.format()) {
            case FMT_FLOAT:  return filterT<float >(src, src_linear, st, st_linear, sigma_d, sigma_r, precision, max_angle, adaptive, order, step_size);
            case FMT_FLOAT3: return filterT<float3>(src, src_linear, st, st_linear, sigma_d, sigma_r, precision, max_angle, adaptive, order, step_size);
            default:
                OZ_INVALID_FORMAT();
        }
    }

}
