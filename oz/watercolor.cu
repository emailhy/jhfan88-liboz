//
// by Daniel Müller and Jan Eric Kyprianidis
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
#include <oz/watercolor.h>
#include <oz/st.h>
#include <oz/oabf.h>
#include <oz/fgauss.h>
#include <oz/blend.h>
#include <oz/color.h>
#include <oz/wog.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>


namespace {
    struct ColorEdge : public oz::generator<float> {
        gpu_sampler<float3,0> src_;

        ColorEdge( const oz::gpu_image& src ) : src_(src) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float3 c = fabs(src_(ix - 1, iy) - src_(ix + 1, iy))
                     + fabs(src_(ix, iy - 1) - src_(ix, iy + 1));
            return (c.x + c.y + c.z) / 3;
        }
    };


    static oz::gpu_image color_edge( const oz::gpu_image& src ) {
        return generate(src.size(), ColorEdge(src));
    }
}


oz::gpu_image oz::watercolor( const gpu_image& src,
                              const gpu_image& st,
                              const gpu_image& noise,
                              float bf_sigma_d, float bf_sigma_r, int bf_N,
                              int nbins, float phi_q,
                              float sigma_c,
                              float nalpha, float nbeta )
{
    gpu_image lfm = st_lfm(st);
    gpu_image lab = rgb2lab(src);

    gpu_image img = lab;
    for (int k = 0; k < bf_N; ++k) {
        img = oabf(img, lfm, bf_sigma_d, bf_sigma_r, true, true, 2);
    }

    img = wog_luminance_quant(img, nbins, phi_q);
    img = lab2rgb(img );

    gpu_image edge = color_edge(src);
    edge = fgauss_filter(edge, st_to_tangent(st), sigma_c);
    edge = invert(edge);

    img = blend_intensity(img, edge, BLEND_LINEAR_BURN);

    gpu_image dst = blend_intensity(img, adjust(noise, nbeta, 1 - nalpha*nbeta),
        BLEND_MULTIPLY, make_float4(1, 1, 1, 1));

    return dst;
}
