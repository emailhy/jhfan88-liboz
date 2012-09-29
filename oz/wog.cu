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
#include <oz/wog.h>
#include <oz/color.h>
#include <oz/bilateral.h>
#include <oz/shuffle.h>
#include <oz/gauss.h>
#include <oz/gpu_sampler2.h>
#include <oz/generate.h>
#include <oz/transform.h>
#include <algorithm>
#include <oz/blend.h>
#include <oz/grad.h>


namespace {
    struct WogDoG : public oz::generator<float> {
        gpu_sampler<float,0> src_;
        oz::gpu_plm2<float> p_;
        float sigma_e_;
        float sigma_r_;
        float tau_;
        float phi_e_;
        float epsilon_;
        float precision_;

        WogDoG( const oz::gpu_image& src, float sigma_e, float sigma_r,
                float tau, float phi_e, float epsilon, float precision )
            : src_(src), sigma_e_(sigma_e), sigma_r_(sigma_r),
              tau_(tau), phi_e_(phi_e), epsilon_(epsilon), precision_(precision), p_(src) {}

        __device__ float operator()( int ix, int iy ) const {
            float twoSigmaE2 = 2.0f * sigma_e_ * sigma_e_;
            float twoSigmaR2 = 2.0f * sigma_r_ * sigma_r_;
            int halfWidth = int(ceilf( precision_ * sigma_r_ ));

            float sumE = 0;
            float sumR = 0;
            float2 norm = make_float2(0);

            for ( int i = -halfWidth; i <= halfWidth; ++i ) {
                for ( int j = -halfWidth; j <= halfWidth; ++j ) {
                    float d = length(make_float2(i,j));
                    float kE = __expf(-d *d / twoSigmaE2);
                    float kR = __expf(-d *d / twoSigmaR2);

                    float c = src_(ix + i, iy + j);
                    sumE += kE * c;
                    sumR += kR * c;
                    norm += make_float2(kE, kR);
                }
            }

            sumE /= norm.x;
            sumR /= norm.y;

            float H = sumE - tau_ * sumR;
            float edge = ( H > epsilon_ )? 1 : 1 + tanhf( phi_e_ * (H - epsilon_) );
            return clamp(edge, 0.0f, 1.0f);
        }
    };

    struct WogLuminanceQuant : public oz::unary_function<float3,float3> {
        int nbins_;
        float phi_q_;

        WogLuminanceQuant( int nbins, float phi_q )
            : nbins_(nbins), phi_q_(phi_q) {}

        __device__ float3 operator()( float3 c ) const {
            float delta_q = 100.0f / nbins_;
            float qn = delta_q * (floor(c.x / delta_q) + 0.5f);
            float qc = qn + 0.5f * delta_q * tanhf(phi_q_ * (c.x - qn));
            return make_float3( qc, c.y, c.z );
        }
    };


    struct WogLuminanceQuant2 : public oz::generator<float3> {
        gpu_sampler<float3,0> src_;
        int nbins_;
        float lambda_delta_;
        float omega_delta_;
        float lambda_phi_;
        float omega_phi_;

        WogLuminanceQuant2( const oz::gpu_image& src, int nbins, float lambda_delta, float omega_delta,
            float lambda_phi, float omega_phi )
            : src_(src), nbins_(nbins), lambda_delta_(lambda_delta), omega_delta_(omega_delta),
              lambda_phi_(lambda_phi), omega_phi_(omega_phi) {}

        __device__ float3 operator()( int ix, int iy ) const {
            float3 c = src_(ix, iy);
            float gx = 0.5f * (src_(ix - 1, iy).x - src_(ix + 1, iy).x);
            float gy = 0.5f * (src_(ix, iy - 1).x - src_(ix, iy + 1).x);
            float grad = sqrtf(gx * gx + gy * gy);
            grad = clamp(grad, lambda_delta_, omega_delta_);
            grad = (grad - lambda_delta_) / (omega_delta_ - lambda_delta_);

            float phi_q = lambda_phi_ + grad * (omega_phi_ - lambda_phi_);
            float delta_q = 100.01f / nbins_;
            float qn = delta_q * (floor(c.x / delta_q) + 0.5f);
            float qc = qn + 0.5f * delta_q * tanhf(phi_q * (c.x - qn));

            return make_float3( qc, c.y, c.z );
        }
    };


    struct WogWarp : public oz::generator<float3> {
        gpu_sampler<float3,0> src_;
        gpu_sampler<float,1> edges_;
        float phi_w_;

        WogWarp( const oz::gpu_image& src, const oz::gpu_image& edges, float phi_w )
            : src_(src, cudaFilterModeLinear), edges_(edges), phi_w_(phi_w) {}

        __device__ float3 operator()( int ix, int iy ) const {
            float gx = 0.5f * (edges_(ix - 1, iy) - edges_(ix + 1, iy));
            float gy = 0.5f * (edges_(ix, iy - 1) - edges_(ix, iy + 1));
            return src_(ix + gx * phi_w_, iy + gy * phi_w_);
        }
    };
}


oz::gpu_image oz::wog_dog( const gpu_image& src, float sigma_e, float sigma_r,
                           float tau, float phi_e, float epsilon, float precision )
{
    return generate(src.size(), WogDoG(src, sigma_e, sigma_r, tau, phi_e, epsilon, precision));
}


oz::gpu_image oz::wog_luminance_quant( const oz::gpu_image& src, int nbins, float phi_q) {
    return transform(src, WogLuminanceQuant(nbins, phi_q));
}


oz::gpu_image oz::wog_luminance_quant( const gpu_image& src, int nbins,
                                       float lambda_delta, float omega_delta,
                                       float lambda_phi, float omega_phi )
{
    return generate(src.size(),
        WogLuminanceQuant2(src, nbins, lambda_delta, omega_delta, lambda_phi, omega_phi));
}


oz::gpu_image oz::wog_warp( const gpu_image& src, const gpu_image& edges, float phi_w) {
    return generate(src.size(), WogWarp(src, edges, phi_w));
}


oz::gpu_image oz::wog_warp_sharp( const gpu_image& src, float sigma_w, float precision_w, float phi_w) {
    gpu_image S = grad_sobel_mag(src);
    S = gauss_filter_xy(S, sigma_w, precision_w);
    return wog_warp(src, S, phi_w);
}


oz::gpu_image oz::wog_abstraction( const gpu_image& src, int n_e, int n_a,
                                   float sigma_d, float sigma_r,
                                   float sigma_e1, float sigma_e2, float precision_e,
                                   float tau, float phi_e, float epsilon,
                                   bool adaptive_quant,
                                   int nbins, float phi_q,
                                   float lambda_delta, float omega_delta,
                                   float lambda_phi, float omega_phi,
                                   bool warp_sharp, float sigma_w,
                                   float precision_w, float phi_w )
{
    gpu_image img;
    gpu_image L;

    {
        img = rgb2lab(src);
        gpu_image E = img;
        gpu_image A = img;

        int N = std::max(n_e, n_a);
        for (int i = 0; i < N; ++i) {
            img = bilateral_filter_xy(img, sigma_d, sigma_r);
            if (i == (n_e - 1)) E = img;
            if (i == (n_a - 1)) A = img;
        }
        img = A;

        L = shuffle(E, 0);
        L = wog_dog( L, sigma_e1, sigma_e2, tau, phi_e, epsilon, precision_e );
    }

    if (adaptive_quant) {
        img = wog_luminance_quant( img, nbins, lambda_delta, omega_delta, lambda_phi, omega_phi );
    } else {
        img = wog_luminance_quant( img, nbins, phi_q );
    }

    img = lab2rgb(img);
    img = blend_intensity(img, L, BLEND_MULTIPLY);

    if (warp_sharp) {
        img = wog_warp_sharp(img, sigma_w, precision_w, phi_w);
    }

    return img;
}
