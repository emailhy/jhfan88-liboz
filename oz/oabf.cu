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
#include <oz/oabf.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>
#include <oz/gpu_plm2.h>


namespace oz {

    template<typename T, bool ustep, bool ustep_x2> struct OabfFilter : public generator<T> {
        gpu_sampler<T,0> src_;
        gpu_plm2<float4> lfm_;
        float sigma_d_;
        float sigma_r_;
        bool tangential_;
        float precision_;

        OabfFilter( const gpu_image& src, const gpu_image lfm, float sigma_d, float sigma_r,
                    bool tangential, bool src_linear, float precision )
            : src_(src, src_linear? cudaFilterModeLinear : cudaFilterModePoint),
              lfm_(lfm), sigma_d_(sigma_d), sigma_r_(sigma_r),
              tangential_(tangential), precision_(precision) {}

        inline __device__ T operator()( int ix, int iy ) const {
            float4 l = lfm_(ix, iy);
            float2 t;
            float sigma_d = sigma_d_;
            if (tangential_) {
                t = make_float2(l.x, l.y);
                sigma_d *= l.z;
            } else {
                t = make_float2(l.y, -l.x);
                sigma_d *= l.w;
            }

            float twoSigmaD2 = 2 * sigma_d * sigma_d;
            float twoSigmaR2 = 2 * sigma_r_ * sigma_r_;
            int halfWidth = int(ceilf( precision_ * sigma_d ));

            float2 tabs = fabs(t);
            float ds;
            float2 dp;
            if (!ustep) {
                ds = 1;
            } else {
                if (tabs.x > tabs.y) {
                    ds = 1.0f / tabs.x;
                    dp = make_float2(0, 0.5f);
                } else {
                    ds = 1.0f / tabs.y;
                    dp = make_float2(0.5f, 0);
                }
            }

            float2 uv = make_float2(0.5f + ix, 0.5f + iy);
            T c0 = src_(uv);
            T sum = c0;

            float norm = 1;
            for (float d = ds; d <= halfWidth; d += ds) {
                float2 dt = d * t;

                if (!ustep_x2) {
                    T c1 = src_(uv + dt);
                    T c2 = src_(uv - dt);

                    T e1 = c1 - c0;
                    T e2 = c2 - c0;

                    float kd = __expf( -dot(d,d) / twoSigmaD2 );
                    float kr1 = __expf( -dot(e1,e1) / twoSigmaR2 );
                    float kr2 = __expf( -dot(e2,e2) / twoSigmaR2 );

                    sum += kd * kr1 * c1;
                    sum += kd * kr2 * c2;
                    norm += kd * kr1 + kd * kr2;
                } else {
                    T c[4];
                    c[0] = src_(uv + dt + dp);
                    c[1] = src_(uv + dt - dp);
                    c[2] = src_(uv - dt + dp);
                    c[3] = src_(uv - dt - dp);

                    float kd = __expf( -dot(d,d) / twoSigmaD2 );
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        T e = c[k] - c0;
                        float kr = __expf( -dot(e,e) / twoSigmaR2 );
                        sum += kd * kr * c[k];
                        norm += kd * kr;
                    }
                }
            }
            return sum / norm;
        }
    };


    template<typename T>
    gpu_image oabf_1dT( const gpu_image& src, const gpu_image& lfm, float sigma_d, float sigma_r,
                       bool tangential, bool src_linear, bool ustep, float precision )
    {
        if (ustep) {
            if (src_linear)
                return generate(src.size(), OabfFilter<T,true,false>(src, lfm, sigma_d, sigma_r, tangential, src_linear, precision));
            else
                return generate(src.size(), OabfFilter<T,true,true>(src, lfm, sigma_d, sigma_r, tangential, src_linear, precision));
        } else {
            return generate(src.size(), OabfFilter<T,false,false>(src, lfm, sigma_d, sigma_r, tangential, src_linear, precision));
        }
    }


    gpu_image oabf_1d( const gpu_image& src, const gpu_image& lfm, float sigma_d, float sigma_r,
                       bool tangential, bool src_linear, bool ustep, float precision )
    {
        switch (src.format()) {
            case FMT_FLOAT:  return oabf_1dT<float >(src, lfm, sigma_d, sigma_r, tangential, src_linear, ustep, precision);
            case FMT_FLOAT3: return oabf_1dT<float3>(src, lfm, sigma_d, sigma_r, tangential, src_linear, ustep, precision);
            default:
                OZ_INVALID_FORMAT();
        }
    }


    gpu_image oabf( const gpu_image& src, const gpu_image& lfm, float sigma_d, float sigma_r,
                    bool src_linear, bool ustep, float precision )
    {
        if (sigma_d <= 0) return src;
        gpu_image dst;
        dst = oabf_1d(src, lfm, sigma_d, sigma_r, false, src_linear, ustep, precision);
        dst = oabf_1d(dst, lfm, sigma_d, sigma_r, true, src_linear, ustep, precision);
        return dst;
    }

}

