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
#include <oz/gmcf.h>
#include <oz/generate.h>
#include <oz/gpu_plm2.h>
#include <oz/gpu_sampler2.h>

namespace oz {

    struct g_pm {
        float lambda_;
        g_pm( float lambda) : lambda_(lambda) {}
        inline __device__ float operator()( float s ) const {
            return 1.0f / (1.0f + s*s / (lambda_ * lambda_));
        }
    };


    struct g_sp {
        float p_;
        g_sp( float p ) : p_(p) {}
        inline __device__ float operator()( float s ) const {
            float d = powf(fabsf(s), p_);
            return (d > 0)? 1.0f /  d : 1e6;
        }
    };


    template <class G> struct imp_gmcf_p1 : public generator<float> {
        gpu_sampler<float,0> u_;
        G g_;

        imp_gmcf_p1( const gpu_image& u, const G& g)
            : u_(u), g_(g) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float u = u_(ix, iy);
            float px = u_(ix + 1, iy) - u;
            float mx = u - u_(ix - 1, iy);
            float py = u_(ix, iy + 1) - u;
            float my = u - u_(ix, iy - 1);

            float grad_u = sqrtf((px*px + mx*mx + py*py + my*my) / 2);
            return g_(grad_u);
        }
    };


    struct imp_gmcf_p2 : public generator<float> {
        gpu_sampler<float,0> u_;
        gpu_sampler<float,1> g_;
        float step_;

        imp_gmcf_p2( const gpu_image& u, const gpu_image& g, float step )
            : u_(u), g_(g), step_(step) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float g = g_(ix, iy);
            float u = u_(ix, iy);
            if (g <= 0) return u;

            float px = u_(ix + 1, iy) - u;
            float mx = u_(ix - 1, iy) - u;
            float py = u_(ix, iy + 1) - u;
            float my = u_(ix, iy - 1) - u;

            float gpx = g_(ix + 1, iy);
            float gmx = g_(ix - 1, iy);
            float gpy = g_(ix, iy + 1);
            float gmy = g_(ix, iy - 1);

            return u + step_ * /*2 **/ (
                gpx * px / (gpx + g) +
                gmx * mx / (gmx + g) +
                gpy * py / (gpy + g) +
                gmy * my / (gmy + g)
            );
        }
    };


    gpu_image gmcf_pm( const gpu_image& src, float lambda, float step ) {
        gpu_image g = generate(src.size(), imp_gmcf_p1<g_pm>(src, g_pm(lambda)));
        return generate(src.size(), imp_gmcf_p2(src, g, step));
    }


    gpu_image gmcf_sp( const gpu_image& src, float p, float step ) {
        gpu_image g = generate(src.size(), imp_gmcf_p1<g_sp>(src, g_sp(p)));
        return generate(src.size(), imp_gmcf_p2(src, g, step));
    }
}
