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
#include <oz/etf.h>
#include <oz/minmax.h>
#include <oz/foreach.h>
#include <oz/generate.h>
#include <oz/gpu_sampler2.h>


namespace {
    struct EtfSobel {
        const gpu_sampler<float,0> src_;
        oz::gpu_plm2<float2> t_;
        oz::gpu_plm2<float> mag_;

        EtfSobel( const oz::gpu_image& src, oz::gpu_image& t, oz::gpu_image& mag )
            : src_(src), t_(t), mag_(mag) {}

        inline __device__ void operator()( int ix, int iy ) {
            float2 g;
            g.x = (
                  -0.183f * src_(ix-1, iy-1) +
                  -0.634f * src_(ix-1, iy) +
                  -0.183f * src_(ix-1, iy+1) +
                  +0.183f * src_(ix+1, iy-1) +
                  +0.634f * src_(ix+1, iy) +
                  +0.183f * src_(ix+1, iy+1)
                  ) * 0.5f;

            g.y = (
                  -0.183f * src_(ix-1, iy-1) +
                  -0.634f * src_(ix,   iy-1) +
                  -0.183f * src_(ix+1, iy-1) +
                  +0.183f * src_(ix-1, iy+1) +
                  +0.634f * src_(ix,   iy+1) +
                  +0.183f * src_(ix+1, iy+1)
                  ) * 0.5f;

            float len = length(g);
            if (len > 0)
                g /= len;

            t_.write(ix, iy, make_float2(-g.y, g.x));
            mag_.write(ix, iy, len);
        }
    };


    struct EtfSmoothFull : public oz::generator<float2> {
        gpu_sampler<float2,0> etf_;
        gpu_sampler<float,1> mag_;
        float sigma_, precision_;
        bool gaussian_;

        EtfSmoothFull( const oz::gpu_image& etf, const oz::gpu_image& mag, float sigma, float precision, bool gaussian )
            : etf_(etf), mag_(mag), sigma_(sigma), precision_(precision), gaussian_(gaussian) {}

        inline __device__ float2 operator()( int ix, int iy ) const {
            int halfWidth = int(ceilf( precision_ * sigma_ ));

            float2 p0 = etf_(ix, iy);
            float z0 = mag_(ix, iy);
            float2 g = make_float2(0);

            for ( int i = -halfWidth; i <= halfWidth; ++i ) {
                for ( int j = -halfWidth; j <= halfWidth; ++j ) {
                    float d = length(make_float2(i,j));
                    if (d <= halfWidth) {
                        float2 p = etf_(ix + i, iy + j);
                        float z = mag_(ix + i, iy + j);

                        float wm = 0.5f * (z - z0 + 1);
                        float wd = dot(p0, p);
                        float w = wm * wd;

                        if (gaussian_) w *= __expf( -0.5f * d *d / sigma_ / sigma_ );

                        g += w * p;
                    }
                }
            }

            float len = length(g);
            if (len > 0)
                g /= len;

            return g;
        }
    };


    template<int dx, int dy> struct EtfSmoothXY : public oz::generator<float2> {
        gpu_sampler<float2,0> etf_;
        gpu_sampler<float,1> mag_;
        float sigma_, precision_;

        EtfSmoothXY( const oz::gpu_image& etf, const oz::gpu_image& mag, float sigma, float precision )
            : etf_(etf), mag_(mag), sigma_(sigma), precision_(precision) {}

        inline __device__ float2 operator()( int ix, int iy ) const {
            int halfWidth = int(ceilf( precision_ * sigma_ ));

            float2 p0 = etf_(ix, iy);
            float z0 = mag_(ix, iy);
            float2 g = make_float2(0);

            for ( int i = -halfWidth; i <= halfWidth; ++i ) {
                float2 p = etf_(ix + dx * i, iy + dy * i);
                float z = mag_(ix + dx * i, iy + dy * i);

                float wm = 0.5f * (z - z0 + 1);
                float wd = dot(p0, p);
                float w = wm * wd;

                g += w * p;
            }

            float len = length(g);
            if (len > 0)
                g /= len;

            return g;
        }
    };
}


oz::gpu_image oz::etf_full( const gpu_image& src, float sigma, int N,
                           float precision, bool gaussian )
{
    gpu_image etf(src.size(), FMT_FLOAT2);
    gpu_image mag(src.size(), FMT_FLOAT);
    {
        EtfSobel f(src, etf, mag);
        foreach(src.w(), src.h(), f);
        float pmax = min(mag);
        if (pmax > 0) mag = mag / pmax;
    }

    for (int k = 0; k < N; ++k) {
        etf = generate(etf.size(), EtfSmoothFull(etf, mag, sigma, precision, gaussian));
    }

    return etf;
}


oz::gpu_image oz::etf_xy( const gpu_image& src, float sigma, int N, float precision ) {
    gpu_image etf(src.size(), FMT_FLOAT2);
    gpu_image mag(src.size(), FMT_FLOAT);
    {
        EtfSobel f(src, etf, mag);
        foreach(src.size(), f);
        float pmax = min(mag);
        if (pmax > 0) mag = mag / pmax;
    }

    for (int k = 0; k < N; ++k) {
        etf = generate(etf.size(), EtfSmoothXY<1,0>(etf, mag, sigma, precision));
        etf = generate(etf.size(), EtfSmoothXY<0,1>(etf, mag, sigma, precision));
    }

    return etf;
}
