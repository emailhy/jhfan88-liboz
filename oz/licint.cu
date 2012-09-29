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
#include <oz/licint.h>
#include <oz/gpu_sampler2.h>
#include <oz/generate.h>
#include <oz/filter_erf.h>


namespace oz {

    template<class SRC, class TF> struct LicIntGaussFlt : generator<float3> {
        const SRC src_;
        const TF tf_;
        float sigma_;
        float precision_;

        LicIntGaussFlt( const SRC& src, const TF& tf, float sigma, float precision )
            : src_(src), tf_(tf), sigma_(sigma), precision_(precision) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            float2 p0 = make_float2(ix + 0.5f, iy + 0.5f);
            filter_erf_gauss<float3,SRC> f(src_, sigma_, precision_);
            lic_integrate<TF,filter_erf_gauss<float3,SRC>,false>( make_float2(ix + 0.5f, iy + 0.5f), tf_, f, src_.w, src_.h);
            return f.result();
        }
    };

    gpu_image licint_gauss_flt( const gpu_image& src, const gpu_image& tf, float sigma, float precision  ) {
        if (sigma <= 0) return src;
        return generate(src.size(), LicIntGaussFlt<gpu_sampler<float3,0>, gpu_sampler<float2,1> >(
            gpu_sampler<float3,0>(src, cudaFilterModeLinear),
            gpu_sampler<float2,1>(tf),
            sigma, precision)
        );
    }


    template<class SRC, class TF> struct LicIntBfFlt : generator<float3> {
        const SRC src_;
        const TF tf_;
        float sigma_d_;
        float sigma_r_;
        float precision_;

        LicIntBfFlt( const SRC& src, const TF& tf, float sigma_d, float sigma_r, float precision )
            : src_(src), tf_(tf), sigma_d_(sigma_d), sigma_r_(sigma_r), precision_(precision) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            float2 p0 = make_float2(ix + 0.5f, iy + 0.5f);
            filter_erf_bf<float3,SRC> f(src_, src_(p0), sigma_d_, sigma_r_, precision_);
            lic_integrate<TF,filter_erf_bf<float3,SRC>,false>( make_float2(ix + 0.5f, iy + 0.5f), tf_, f, src_.w, src_.h);
            return f.result();
        }
    };

    gpu_image licint_bf_flt( const gpu_image& src, const gpu_image& tf,
                             float sigma_d, float sigma_r, float precision )
    {
        if (sigma_d <= 0) return src;
        return generate(src.size(), LicIntBfFlt<gpu_sampler<float3,0>, gpu_sampler<float2,1> >(
            gpu_sampler<float3,0>(src, cudaFilterModeLinear),
            gpu_sampler<float2,1>(tf),
            sigma_d, sigma_r, precision)
        );
    }

}
