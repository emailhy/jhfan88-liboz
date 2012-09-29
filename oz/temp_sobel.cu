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
#include <oz/temp_sobel.h>
#include <oz/generate.h>
#include <oz/gpu_sampler3.h>

namespace oz {

    struct imp_temp_sobel : public generator<float3> {
        gpu_sampler<float,0> p_;
        gpu_sampler<float,1> c_;
        gpu_sampler<float,2> n_;

        imp_temp_sobel( const gpu_image& p, const gpu_image& c, const gpu_image& n )
            : p_(p), c_(c), n_(n) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            float gx =
               (1 * p_(ix+1, iy-1) +
                2 * p_(ix+1, iy  ) +
                1 * p_(ix+1, iy+1) +
                2 * c_(ix+1, iy-1) +
                4 * c_(ix+1, iy  ) +
                2 * c_(ix+1, iy+1) +
                1 * n_(ix+1, iy-1) +
                2 * n_(ix+1, iy  ) +
                1 * n_(ix+1, iy+1))
                -
               (1 * p_(ix-1, iy-1) +
                2 * p_(ix-1, iy  ) +
                1 * p_(ix-1, iy+1) +
                2 * c_(ix-1, iy-1) +
                4 * c_(ix-1, iy  ) +
                2 * c_(ix-1, iy+1) +
                1 * n_(ix-1, iy-1) +
                2 * n_(ix-1, iy  ) +
                1 * n_(ix-1, iy+1));

            float gy =
               (1 * p_(ix-1, iy+1) +
                2 * p_(ix  , iy+1) +
                1 * p_(ix+1, iy+1) +
                2 * c_(ix-1, iy+1) +
                4 * c_(ix  , iy+1) +
                2 * c_(ix+1, iy+1) +
                1 * n_(ix-1, iy+1) +
                2 * n_(ix  , iy+1) +
                1 * n_(ix+1, iy+1))
                -
               (1 * p_(ix-1, iy-1) +
                2 * p_(ix  , iy-1) +
                1 * p_(ix+1, iy-1) +
                2 * c_(ix-1, iy-1) +
                4 * c_(ix  , iy-1) +
                2 * c_(ix+1, iy-1) +
                1 * n_(ix-1, iy-1) +
                2 * n_(ix  , iy-1) +
                1 * n_(ix+1, iy-1));

            float gz =
               (1 * n_(ix-1, iy-1) +
                2 * n_(ix  , iy-1) +
                1 * n_(ix+1, iy-1) +
                2 * n_(ix-1, iy  ) +
                4 * n_(ix  , iy  ) +
                2 * n_(ix+1, iy  ) +
                1 * n_(ix-1, iy+1) +
                2 * n_(ix  , iy+1) +
                1 * n_(ix+1, iy+1));

            return make_float3(gx/16, gy/16, gz/16);
        }
    };
}

oz::gpu_image oz::temp_sobel( const gpu_image& p, const gpu_image& c, const gpu_image& n ) {
    if ((p.size() != c.size()) ||
        (p.size() != n.size())) OZ_INVALID_SIZE();
    return generate(p.size(), imp_temp_sobel(p, c, n));
}
