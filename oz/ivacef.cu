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
#include <oz/ivacef.h>
#include <oz/color.h>
#include <oz/st.h>
#include <oz/st_util.h>
#include <oz/gauss.h>
#include <oz/stgauss3.h>
#include <oz/generate.h>
#include <oz/gpu_sampler2.h>
#include <oz/gpu_plm2.h>
#include <oz/shuffle.h>

namespace oz {

    struct IvacefSobel : public generator<float4> {
        gpu_sampler<float3,0> src_;
        gpu_plm2<float3> prev_;
        float threshold2_;

        IvacefSobel( const gpu_image& src, const gpu_image& prev, float threshold )
            : src_(src), prev_(prev.is_valid()? prev : gpu_plm2<float3>()), threshold2_(threshold * threshold) {}

        inline __device__ float4 operator()( int ix, int iy ) const {
            float3 u = (
                  -0.183f * src_(ix - 1, iy - 1) +
                  -0.634f * src_(ix - 1, iy    ) +
                  -0.183f * src_(ix - 1, iy + 1) +
                  +0.183f * src_(ix + 1, iy - 1) +
                  +0.634f * src_(ix + 1, iy    ) +
                  +0.183f * src_(ix + 1, iy + 1)
                  ) * 0.5f;

            float3 v = (
                  -0.183f * src_(ix - 1, iy - 1) +
                  -0.634f * src_(ix,     iy - 1) +
                  -0.183f * src_(ix + 1, iy - 1) +
                  +0.183f * src_(ix - 1, iy + 1) +
                  +0.634f * src_(ix,     iy + 1) +
                  +0.183f * src_(ix + 1, iy + 1)
                  ) * 0.5f;

            float3 g = make_float3(dot(u, u), dot(v, v), dot(u, v));

            float mag = g.x * g.x + g.y * g.y + 2 * g.z * g.z;
            if (mag < threshold2_) {
                if (prev_.ptr) {
                    return make_float4(prev_(ix, iy), 1);
                } else {
                    return make_float4( g, 0 );
                }
            } else {
                return make_float4(g, 1);
            }
        }
    };


    gpu_image ivacef_sobel( const gpu_image& src, const gpu_image& prev, float tau_r ) {
        return generate(src.size(), IvacefSobel(src, prev, tau_r));
    }


    struct JacobiStep : public generator<float4> {
        gpu_sampler<float4,0> src_;

        JacobiStep( const gpu_image& src ) : src_(src) {}

        inline __device__ float4 operator()( int ix, int iy )const {
#if 0
            float4 o = src_(ix, iy);
            if (o.w < 1) {
                o = make_float4( 0.25f * (
                    make_float3(src_(ix,   iy+1)) +
                    make_float3(src_(ix-1, iy  )) +
                    make_float3(src_(ix+1, iy  )) +
                    make_float3(src_(ix  , iy-1))), 0);
            }
            return o;
#else
            float4 c = src_(ix, iy);
            float3 o;
            if (c.w > 0) {
                o = make_float3(c);
            } else {
                o = 0.25f * (
                   make_float3(src_(ix+1, iy)) +
                   make_float3(src_(ix-1, iy)) +
                   make_float3(src_(ix, iy+1)) +
                   make_float3(src_(ix, iy-1))
                );
            }

            return make_float4( o, c.w );
#endif
        }
    };


    struct RelaxDown : public generator<float4> {
        gpu_sampler<float4,0> src_;

        RelaxDown( const gpu_image& src ) : src_(src) {}

        inline __device__ float4 operator()( int ix, int iy )const {
#if 0
            int i = 2*ix;
            int j = 2*iy;
            float4 sum = make_float4(0);
            float4 c;
            c = src_(i,   j  ); if (c.w == 1) { sum += c; }
            c = src_(i+1, j  ); if (c.w == 1) { sum += c; }
            c = src_(i  , j+1); if (c.w == 1) { sum += c; }
            c = src_(i+1, j+1); if (c.w == 1) { sum += c; }
            if (sum.w > 0) sum /= sum.w;
            return sum;
#else
            float4 sum = make_float4(0);
            {
                float4 c = src_(2*ix, 2*iy);
                if (c.w > 0) sum += make_float4(make_float3(c), 1);
            }

            if (2*ix+1 < src_.w) {
                float4 c = src_(2*ix+1, 2*iy);
                if (c.w > 0) sum += make_float4(make_float3(c), 1);
            }

            if (2*iy+1 < src_.h) {
                float4 c = src_(2*ix, 2*iy+1);
                if (c.w > 0) sum += make_float4(make_float3(c), 1);

                if (2*ix+1 < src_.w) {
                    float4 c = src_(2*ix+1, 2*iy+1);
                    if (c.w > 0) sum += make_float4(make_float3(c), 1);
                }
            }

            if (sum.w > 0) {
                return make_float4(make_float3(sum) / sum.w, 1);
            } else {
                return make_float4(0);
            }
#endif
        }
    };


    struct RelaxUp : public generator<float4> {
        gpu_plm2<float4> src0_;
        gpu_sampler<float4,0> src1_;

        RelaxUp( const gpu_image& src0, const gpu_image& src1 )
            : src0_(src0), src1_(src1, cudaFilterModeLinear) {}

        inline __device__ float4 operator()( int ix, int iy )const {
            float4 c = src0_(ix, iy);
            if (c.w < 1) {
                float2 uv = make_float2(0.5f * (ix + 0.5f), 0.5f * (iy + 0.5f));
                c = make_float4(make_float3(src1_(uv.x, uv.y)), 0);
            }
            return c;
        }
    };


    static gpu_image jacobi_step( const gpu_image& src ) {
        return generate(src.size(), JacobiStep(src));
    }

    static gpu_image relax_down( const gpu_image& src ) {
        log_image(src.convert(FMT_FLOAT3), "RA");
        gpu_image R = generate((src.w()+1)/2, (src.h()+1)/2, RelaxDown(src));
        log_image(R.convert(FMT_FLOAT3), "RD");
        return R;
    }

    static gpu_image relax_up( const gpu_image& src0, const gpu_image& src1 ) {
        return generate(src0.size(), RelaxUp(src0, src1));
    }

    gpu_image ivacef_relax( const gpu_image& st, int v2 ) {
        if ((st.w() <= 2) || (st.h() <= 2)) return st;
        gpu_image tmp;
        tmp = relax_down(st);
        tmp = ivacef_relax(tmp, v2);
        tmp = relax_up(st, tmp);
        for (int k = 0; k < v2; ++k) tmp = jacobi_step(tmp);
        return tmp;
    }


    gpu_image ivacef_compute_st( const gpu_image& src, const gpu_image& prev,
                                         float sigma_d, float tau_r, int v2 )
    {
        gpu_image st = ivacef_sobel(src, prev, tau_r);
        if (!prev.is_valid()) {
            st = ivacef_relax(st, v2);
        }
        st = st.convert(FMT_FLOAT3);
        return gauss_filter_xy(st, sigma_d);
    }


    struct IvacefSign : public generator<float> {
        gpu_sampler<float,0> L_;
        gpu_plm2<float3> st_;
        float sigma_;
        float tau_;

        IvacefSign( const gpu_image& L, const gpu_image& st, float sigma, float tau )
            : L_(L,cudaFilterModeLinear), st_(st), sigma_(sigma), tau_(tau) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float2 n = st2gradient(st_(ix, iy));
            float2 nabs = fabs(n);
            float ds = 1.0f / ((nabs.x > nabs.y)? nabs.x : nabs.y);

            float halfWidth = 5 * sigma_;
            float sigma2 = sigma_ * sigma_;
            float twoSigma2 = 2 * sigma2;

            float sum = -sigma2 * L_(ix + 0.5f, iy + 0.5f);
            for( float d = ds; d <= halfWidth; d += ds ) {
                float k = (d*d - sigma2) * __expf( -d*d / twoSigma2 );
                float2 o = d*n;
                float c = L_(0.5f + ix - o.x, 0.5f + iy - o.y) +
                          L_(0.5f + ix + o.x, 0.5f + iy + o.y);
                sum += k * c;
            }

            sum = sum / (sqrtf(2*CUDART_PI_F) * sigma2 * sigma_);
            if (fabs(sum) < tau_) sum = 0;
            return sum;
        }
    };


    gpu_image ivacef_sign( const gpu_image& L, const gpu_image& st, float sigma, float tau ) {
        return generate(L.size(), IvacefSign(L, st, sigma, tau));
    }



    struct minmax_gray_t {
        float max_sum;
        float min_sum;
        float3 max_val;
        float3 min_val;
        float gray_sum;
        float gray_N;

        __device__ void init(float3 c) {
            float sum = 0.299f * abs(c.z) + 0.587f * abs(c.y) + 0.114f * abs(c.x);
            max_val = min_val = c;
            max_sum = min_sum = gray_sum = sum;
            gray_N = 1;
        }

        __device__ void add(float3 c) {
            float sum = 0.299f * abs(c.z) + 0.587f * abs(c.y) + 0.114f * abs(c.x);
            gray_sum += sum;
            gray_N += 1;
            if (sum > max_sum) {
                max_sum = sum;
                max_val = c;
            }
            if (sum < min_sum) {
                min_sum = sum;
                min_val = c;
            }
        }

        //__device__ float gray_mean() { return gray_sum / gray_N; }
    };


    struct IvacefShock : public generator<float3> {
        gpu_sampler<float3,1> src_;
        gpu_plm2<float3> st_;
        gpu_plm2<float> sign_;
        float radius_;

        IvacefShock( const gpu_image& src, const gpu_image& st, const gpu_image& sign, float radius )
            : src_(src), st_(st), sign_(sign), radius_(radius) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            float2 n = st2gradient(st_(ix, iy));
            float2 nabs = fabs(n);
            float sign = sign_(ix, iy);

            minmax_gray_t mm;
            float3 c0 = src_(ix + 0.5f, iy + 0.5f);
            mm.init(c0);
            if (dot(n,n) > 0) {
                float ds;
                float2 dp;
                if (nabs.x > nabs.y) {
                    ds = 1.0f / nabs.x;
                    dp = make_float2(0, 0.5f);
                } else {
                    ds = 1.0f / nabs.y;
                    dp = make_float2(0.5f, 0);
                }

                float2 uv = make_float2(ix + 0.5f, iy + 0.5f);
                for( float d = ds; d <= radius_; d += ds ) {
                    float2 o = d*n;
                    float2 q;

                    q = make_float2(uv.x + o.x + dp.x, uv.y + o.y + dp.y);
                    {
                        float3 c = src_(q.x, q.y);
                        mm.add(c);
                    }

                    q = make_float2(uv.x + o.x - dp.x, uv.y + o.y - dp.y);
                    {
                        float3 c = src_(q.x, q.y);
                        mm.add(c);
                    }

                    q = make_float2(uv.x - o.x + dp.x, uv.y - o.y + dp.x);
                    {
                        float3 c = src_(q.x, q.y);
                        mm.add(c);
                    }

                    q = make_float2(uv.x - o.x - dp.x, uv.y - o.y - dp.x);
                    {
                        float3 c = src_(q.x, q.y);
                        mm.add(c);
                    }
                }
            }

            return (sign < 0)? mm.max_val : ((sign > 0)? mm.min_val : c0);
        }
    };


    gpu_image ivacef_shock( const gpu_image& src, const gpu_image& st, const gpu_image& sign, float radius ) {
        return generate(src.size(), IvacefShock(src, st, sign, radius));
    }


    gpu_image ivacef( const gpu_image& src, int N,
                              float sigma_d, float tau_r, int v2, float sigma_t,
                              float max_angle, float sigma_i, float sigma_g,
                              float r, float tau_s, float sigma_a, bool adaptive )
    {
        if (src.format() != FMT_FLOAT3) OZ_INVALID_FORMAT();
        gpu_image img = src;
        gpu_image st;

        for (int k = 0; k < N; ++k) {
            st = ivacef_compute_st(img, st, sigma_d, tau_r, v2);
            img = stgauss3_filter_(img, st, sigma_t, true, true, true, 2, 1);

            st = ivacef_compute_st(img, st, sigma_d, tau_r, v2);
            gpu_image L = gauss_filter_xy(rgb2gray(img), sigma_i);
            gpu_image sign = ivacef_sign(L, st, sigma_g, tau_s);
            img = ivacef_shock(src, st, sign, r);
        }

        img = stgauss3_filter_(img, st, sigma_a, true, true, false, 2, 1);
        return img;
    }

}
