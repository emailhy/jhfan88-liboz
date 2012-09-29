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
#include <oz/beltrami.h>
#include <oz/foreach.h>
#include <oz/generate.h>
#include <oz/gpu_sampler2.h>
#include <oz/gpu_plm2.h>


namespace oz {
    struct BeltramiP1 {
        const gpu_sampler<float3,0> src_;
        const float beta_;
        gpu_plm2<float> g_;
        gpu_plm2<float3> px_;
        gpu_plm2<float3> py_;

        BeltramiP1( const gpu_image& src, float beta,
                    gpu_image g, gpu_image px, gpu_image py )
            : src_(src), beta_(beta), g_(g), px_(px), py_(py) {}

        inline __device__ void operator()( int ix, int iy ) {
            float3 sx = src_(ix, iy) - src_(ix - 1, iy);
            float3 sy = src_(ix, iy) - src_(ix, iy - 1);

            float4 g;
            g.x = beta_ + dot(sx, sx);
            g.y = beta_ + dot(sy, sy);
            g.z = dot(sx, sy);
            g.w = rsqrtf(g.x * g.y - g.z*g.z);

            float3 px = g.w * (  g.y * sx - g.z * sy);
            float3 py = g.w * (- g.z * sx + g.x * sy);

            g_.write(ix, iy, g.w);
            px_.write(ix, iy, px);
            py_.write(ix, iy, py);
        }
    };

    struct BeltramiP2 : public generator<float3> {
        gpu_plm2<float3> src_;
        gpu_sampler<float3,0> px_;
        gpu_sampler<float3,1> py_;
        gpu_plm2<float> g_;
        float dt_;

        BeltramiP2( const gpu_image& src, const gpu_image& px,
                    const gpu_image& py, const gpu_image& g, float dt)
            : src_(src), px_(px), py_(py), g_(g), dt_(dt) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            float gm05 = g_(ix,iy);

            float3 c = src_(ix, iy);
            float3 dpx = px_(ix+1, iy) - px_(ix, iy);
            float3 dpy = py_(ix, iy+1) - py_(ix, iy);

            return c + dt_ * gm05 * (dpx + dpy);
        }
    };
}


oz::gpu_image oz::beltrami( const gpu_image& src, float beta, float step ) {
    gpu_image g(src.size(), FMT_FLOAT);
    gpu_image px(src.size(), FMT_FLOAT3);
    gpu_image py(src.size(), FMT_FLOAT3);
    BeltramiP1 f(src, beta, g, px, py);
    foreach(src.size(), f);
    return generate(src.size(), BeltramiP2(src, px, py, g, step * beta));
}
