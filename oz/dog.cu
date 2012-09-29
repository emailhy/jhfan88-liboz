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
#include <oz/dog.h>
#include <oz/generate.h>
#include <oz/transform.h>
#include <oz/gpu_sampler1.h>


namespace oz {

    struct imp_dog_filter : public generator<float> {
        gpu_sampler<float,0> src_;
        float sigma_e_, sigma_r_;
        float tau0_, tau1_;
        float precision_;

        imp_dog_filter( const gpu_image& src, float sigma_e, float sigma_r,
                        float tau0, float tau1, float precision )
            : src_(src), sigma_e_(sigma_e), sigma_r_(sigma_r),
              tau0_(tau0), tau1_(tau1), precision_(precision) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float twoSigmaE2 = 2 * sigma_e_ * sigma_e_;
            float twoSigmaR2 = 2 * sigma_r_ * sigma_r_;
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

            float H = sumE - tau0_ * sumR;
            if (fabs(H) < tau1_) H = 0;

            return H;
        }
    };

    gpu_image dog_filter( const gpu_image& src, float sigma_e, float sigma_r,
                          float tau0, float tau1, float precision )
    {
        return generate(src.size(), imp_dog_filter(src, sigma_e, sigma_r, tau0, tau1, precision));
    }


    struct imp_dog_threshold_tanh : public unary_function<float,float> {
        float epsilon_, phi_;

        imp_dog_threshold_tanh( float epsilon, float phi )
            : epsilon_(epsilon), phi_(phi) {}

        inline __device__ float operator()( float H ) const {
            //float edge = ( H > epsilon )? 1 : 1 + tanh( phi * H);
            //float edge = ( H < epsilon )? 0 : tanh( phi * (H - epsilon));
            float edge = ( H > epsilon_ )? 1 : 1 + tanh( phi_ * (H - epsilon_));
            return ::clamp(edge, 0.0f, 1.0f);
        }
    };

    gpu_image dog_threshold_tanh( const gpu_image& src, float epsilon, float phi ) {
        if (phi <= 0) return src;
        return transform(src, imp_dog_threshold_tanh(epsilon, phi));
    }


    struct imp_dog_colorize : public unary_function<float,float3> {
        inline __device__ float3 operator()( float s ) const {
            float H = ::clamp(s, -1.0f, 1.0f);
            return make_float3( 0, (H > 0)? H : 0, (H < 0)? -H : 0);
        }
    };

    gpu_image dog_colorize( const gpu_image& src ) {
        return transform(src, imp_dog_colorize());
    }


    struct imp_dog_sign : public unary_function<float,float> {
        inline __device__ float operator()( float H ) const {
            return (H > 0)? 1 : 0;
        }
    };

    gpu_image dog_sign( const gpu_image& src ) {
        return transform(src, imp_dog_sign());
    }


    struct imp_gradient_dog : public generator<float> {
        const gpu_sampler<float,0> src_;
        const gpu_plm2<float2> tm_;
        float sigma_e_;
        float sigma_r_;
        float tau0_;
        float tau1_;
        float precision_;

        imp_gradient_dog( const gpu_image& src, const gpu_image& tm, float sigma_e, float sigma_r,
                          float tau0, float tau1, float precision)
            : src_(src), tm_(tm), sigma_e_(sigma_e), sigma_r_(sigma_r),
              tau0_(tau0), tau1_(tau1), precision_(precision) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float2 t = tm_(ix, iy);
            float2 n = make_float2(t.y, -t.x);
            float2 nabs = fabs(n);
            float ds = 1.0f / ((nabs.x > nabs.y)? nabs.x : nabs.y);

            float twoSigmaE2 = 2 * sigma_e_ * sigma_e_;
            float twoSigmaR2 = 2 * sigma_r_ * sigma_r_;
            float halfWidth = precision_ * sigma_r_;

            float sumE = src_(ix, iy);
            float sumR = sumE;
            float2 norm = make_float2(1, 1);

            for( float d = ds; d <= halfWidth; d += ds ) {
                float kE = __expf( -d * d / twoSigmaE2 );
                float kR = __expf( -d * d / twoSigmaR2 );

                float2 o = d*n;
                float c = src_(0.5f + ix - o.x, 0.5f + iy - o.y) +
                          src_(0.5f + ix + o.x, 0.5f + iy + o.y);
                sumE += kE * c;
                sumR += kR * c;
                norm += 2 * make_float2(kE, kR);
            }
            sumE /= norm.x;
            sumR /= norm.y;

            float H = sumE - tau0_ * sumR;
            if (fabs(H) < tau1_) H = 0;

            return H;
        }
    };

    gpu_image gradient_dog( const gpu_image& src, const gpu_image& tm, float sigma_e, float sigma_r,
                            float tau0, float tau1, float precision )
    {
        return generate(src.size(), imp_gradient_dog(src, tm, sigma_e, sigma_r, tau0, tau1, precision));
    }


    struct imp_gradient_log : public generator<float> {
        const gpu_sampler<float,0> src_;
        const gpu_plm2<float2> tm_;
        float sigma_;
        float tau_;

        imp_gradient_log( const gpu_image& src, const gpu_image& tm, float sigma, float tau )
            : src_(src), tm_(tm), sigma_(sigma), tau_(tau) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float sigma2 = sigma_ * sigma_;
            float twoSigma2 = 2 * sigma2;

            float2 t = tm_(ix, iy);
            float2 n = make_float2(t.y, -t.x);
            float2 nabs = fabs(n);
            float ds = 1.0f / ((nabs.x > nabs.y)? nabs.x : nabs.y);

            float sum = -sigma2 * src_(ix + 0.5f, iy + 0.5f);

            float halfWidth = 5 * sigma_;
            for( float d = ds; d <= halfWidth; d += ds ) {
                float k = (d*d - sigma2) * __expf( -d*d / twoSigma2 );

                float2 o = d*n;
                float c = src_(0.5f + ix - o.x, 0.5f + iy - o.y) +
                          src_(0.5f + ix + o.x, 0.5f + iy + o.y);
                sum += k * c;
            }

            sum = -sum / (sqrtf(2*CUDART_PI_F) * sigma2 * sigma_);
            if (fabs(sum) < tau_) sum = 0;
            return sum;
        }
    };


    gpu_image gradient_log( const gpu_image& src, const gpu_image& tm, float sigma, float tau ) {
        return generate(src.size(), imp_gradient_log(src, tm, sigma, tau));
    }
}
