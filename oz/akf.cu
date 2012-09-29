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
#include <oz/akf.h>
#include <oz/foreach.h>
#include <oz/gpu_plm2.h>
#include <oz/gpu_sampler3.h>
#include <oz/st_util.h>


namespace oz {

    template<bool debug, typename T, int N> struct imp_akf {
        gpu_plm2<T> dst_;
        gpu_sampler<T,0> src_;
        gpu_sampler<float3,1> st_;
        gpu_sampler<float,2> krnl_;
        float radius_;
        float alpha_;
        float threshold_;
        float q_;
        float a_star_;
        gpu_plm2<float> w_;

        imp_akf( gpu_image dst, const gpu_image& src, const gpu_image& st, const gpu_image& krnl,
                 float radius, float q, float alpha, float threshold, float a_star, gpu_image *w )
            : dst_(dst), src_(src), st_(st), krnl_(krnl,cudaFilterModeLinear, cudaAddressModeWrap, true),
              radius_(radius), q_(q), alpha_(alpha), a_star_(a_star), threshold_(threshold)
        {
            if (debug) {
                if (!w || (w->w() != N * src.w()) || (w->h() != src.h())) OZ_X() << "Invalid weight buffer!";
                w_ = gpu_plm2<float>(*w);
            }
        }

        inline __device__ void operator()( int ix, int iy ) {
            float3 t = st2tA(st_(ix, iy), a_star_);
            float a = radius_ * clamp((alpha_ + t.z) / alpha_, 0.1f, 2.0f);
            float b = radius_ * clamp(alpha_ / (alpha_ + t.z), 0.1f, 2.0f);
            int max_x = int(sqrtf(a*a * t.x*t.x + b*b * t.y*t.y));
            int max_y = int(sqrtf(a*a * t.y*t.y + b*b * t.x*t.x));
            float4 SR = 0.5f * make_float4(t.x/a, -t.y/b, t.y/a, t.x/b);

            float piN = 2 * CUDART_PI_F / N;
            float4 RpiN = make_float4(cosf(piN), sinf(piN), -sinf(piN), cosf(piN));

            T m[N];
            T s[N];
            float w[N];
            for (int k = 0; k < N; ++k) {
                m[k] = make_zero<T>();
                s[k] = make_zero<T>();
                w[k] = 0;
            }

            for (int j = -max_y; j <= max_y; ++j) {
                for (int i = -max_x; i <= max_x; ++i) {
                    float2 v = make_float2( SR.x * i + SR.z * j,
                                            SR.y * i + SR.w * j );

                    if (dot(v,v) <= 0.25f) {
                        T c = src_(ix + i, iy + j);
                        T cc = c * c;

                        for (int k = 0; k < N; ++k) {
                            float wx = krnl_(v);

                            m[k] += c * wx;
                            s[k] += cc * wx;
                            w[k] += wx;

                            v = make_float2( RpiN.x * v.x + RpiN.z * v.y,
                                             RpiN.y * v.x + RpiN.w * v.y );
                        }
                    }
                }
            }

            T o = make_zero<T>();
            float ow = 0;
            for (int k = 0; k < N; ++k) {
                m[k] /= w[k];
                s[k] = fabs(s[k] / w[k] - m[k] * m[k]);
                float sigma2 = fmaxf(threshold_, sqrtf(sum(s[k])));
                float wk = __powf(sigma2, -q_);
                o += m[k] * wk;
                ow += wk;
                if (debug) {
                    w_.write(N * ix + k, iy, wk);
                }
            }

            dst_.write(ix, iy, o / ow);
        }
    };


    template <typename T>
    gpu_image akf_filterT( const gpu_image& src, const gpu_image& st,
                           const gpu_image& krnl, float radius, float q,
                           float alpha, float threshold, float a_star, int N, gpu_image *w )
    {
        gpu_image dst(src.size(), src.format());
        if (N == 4) {
            if (w) {
                imp_akf<true,T,4> f(dst, src, st, krnl, radius, q, alpha, threshold, a_star, w);
                foreach(src.size(), f);
            } else {
                imp_akf<false,T,4> f(dst, src, st, krnl, radius, q, alpha, threshold, a_star, w);
                foreach(src.size(), f);
            }
        } else {
            if (w) {
                imp_akf<true,T,8> f(dst, src, st, krnl, radius, q, alpha, threshold, a_star, w);
                foreach(src.size(), f);
            } else {
                imp_akf<false,T,8> f(dst, src, st, krnl, radius, q, alpha, threshold, a_star, w);
                foreach(src.size(), f);
            }
        }
        return dst;
    }


    gpu_image akf_filter( const gpu_image& src, const gpu_image& st,
                          const gpu_image& krnl, float radius, float q,
                          float alpha, float threshold, float a_star, int N, gpu_image *w )
    {
        if ((N != 4) && (N != 8)) OZ_X() << "Invalid N!";
        switch (src.format()) {
            case FMT_FLOAT:  return akf_filterT<float >(src, st, krnl, radius, q, alpha, threshold, a_star, N, w);
            case FMT_FLOAT3: return akf_filterT<float3>(src, st, krnl, radius, q, alpha, threshold, a_star, N, w);
            default:
                OZ_INVALID_FORMAT();
        }
    }

}
