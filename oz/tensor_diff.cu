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
#include <oz/tensor_diff.h>
#include <oz/generate.h>
#include <oz/gpu_sampler2.h>
#include <oz/gauss.h>
#include <oz/st.h>
#include <oz/st_util.h>

namespace oz {

    template<typename T> struct TensorDiff : public generator<T> {
        const gpu_sampler<T,0> u_;
        const gpu_sampler<float3,1> D_;
        float dt_;

        TensorDiff( const gpu_image& u, const gpu_image& D, float dt )
            : u_(u), D_(D), dt_(dt) {}

        inline __device__ T operator()( int ix, int iy ) const {
            T ump = u_(ix - 1, iy + 1);
            T ucp = u_(ix,     iy + 1);
            T upp = u_(ix + 1, iy + 1);
            T umc = u_(ix - 1, iy    );
            T ucc = u_(ix,     iy    );
            T upc = u_(ix + 1, iy    );
            T umm = u_(ix - 1, iy - 1);
            T ucm = u_(ix,     iy - 1);
            T upm = u_(ix + 1, iy - 1);

            float3 Dcp = D_(ix,     iy + 1);
            float3 Dmc = D_(ix - 1, iy    );
            float3 Dcc = D_(ix,     iy    );
            float3 Dpc = D_(ix + 1, iy    );
            float3 Dcm = D_(ix,     iy - 1);

            return ucc + dt_ * (
                -0.25f * (Dmc.z + Dcp.z) * ump
                 +0.5f * (Dcp.y + Dcc.y) * ucp
                +0.25f * (Dpc.z + Dcp.z) * upp
                 +0.5f * (Dmc.x + Dcc.x) * umc
                 -0.5f * (Dmc.x + 2*Dcc.x + Dpc.x + Dcm.y + 2*Dcc.y + Dcp.y) * ucc
                 +0.5f * (Dpc.x + Dcc.x) * upc
                +0.25f * (Dmc.z + Dcm.z) * umm
                 +0.5f * (Dcm.y + Dcc.y) * ucm
                -0.25f * (Dpc.z + Dcm.z) * upm
            );
        }
    };


    static inline __device__ float weickert_g( float s, float Cm, float lambda, int m ) {
        if (s <= 0) return 1;
        return 1.0f - exp((double)-Cm / ::pow((double)s / lambda, m));
    }


    static inline __device__ float3 diffusion_tensor( float3 J, float lambda1, float lambda2 ) {
        float2 g = st2gradient(J);
        float a = lambda1 * g.x * g.x + lambda2 * g.y * g.y;
        float b = (lambda1 - lambda2) * g.x * g.y;
        float c = lambda1 * g.y * g.y + lambda2 *g.x * g.x;
        return make_float3(a, c, b);
    }


    struct EE_Tensor : public generator<float3> {
        const gpu_plm2<float3> J_;
        float lambda_;

        EE_Tensor( const gpu_image& J, float lambda )
            : J_(J), lambda_(lambda) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            float3 J = J_(ix, iy);
            float2 mu = st2lambda(J);

            float lambda1 = weickert_g(mu.x, 3.31488f, lambda_*lambda_, 4);
            float lambda2 = 1;

            return diffusion_tensor(J, lambda1, lambda2);
        }
    };


    gpu_image ee_diff( const gpu_image& src, float sigma, float rho,
                       float lambda, float dt, int N )
    {
        gpu_image img = src;
        gpu_image img_smooth;
        for (int k = 0; k < N; ++k) {
            img_smooth = img;
            if (sigma > 0) img_smooth = gauss_filter_xy(img_smooth, sigma, 3);
            gpu_image st = st_scharr_3x3(img_smooth);
            if (rho) st = gauss_filter_xy(st, rho, 3);
            gpu_image D = generate(img.size(), EE_Tensor(st, lambda));
            switch (img.format()) {
                case FMT_FLOAT:  img = generate(img.size(), TensorDiff<float >(img, D, dt)); break;
                case FMT_FLOAT3: img = generate(img.size(), TensorDiff<float3>(img, D, dt)); break;
                default:
                    OZ_INVALID_FORMAT();
            }
        }
        return img;
    }


    struct CE_Tensor : public generator<float3> {
        const gpu_plm2<float3> J_;
        float alpha_;
        float C_;
        int m_;

        CE_Tensor( const gpu_image& J, float alpha, float C, int m )
            : J_(J), alpha_(alpha), C_(C), m_(m) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            float3 J = J_(ix, iy);
            float2 mu = st2lambda(J);

            float lambda1 = alpha_;
            float lambda2 = (double)alpha_ + (1.0 - (double)alpha_) * exp((double)-C_ / ::pow((double)(mu.x - mu.y), 2*m_));

            return diffusion_tensor(J, lambda1, lambda2);
        }
    };


    gpu_image ce_diff( const gpu_image& src, float sigma, float rho,
                       float alpha, float C, int m, float dt, int N )
    {
        gpu_image img = src;
        gpu_image img_smooth;
        for (int k = 0; k < N; ++k) {
            img_smooth = img;
            if (sigma > 0) img_smooth = gauss_filter_xy(img_smooth, sigma, 3);
            gpu_image st = st_scharr_3x3(img_smooth);
            if (rho) st = gauss_filter_xy(st, rho, 3);
            gpu_image D = generate(img.size(), CE_Tensor(st, alpha, C, m));
            switch (img.format()) {
                case FMT_FLOAT:  img = generate(img.size(), TensorDiff<float >(img, D, dt)); break;
                case FMT_FLOAT3: img = generate(img.size(), TensorDiff<float3>(img, D, dt)); break;
                default:
                    OZ_INVALID_FORMAT();
            }
        }
        return img;
    }
}
