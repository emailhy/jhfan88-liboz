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
#include <oz/licint.h>
#include <oz/cpu_sampler.h>
#include <oz/filter_path.h>


namespace oz {

    std::vector<float3> oz::licint_path( int ix, int iy, const cpu_image& tf, float sigma, float precision, bool midpoint ) {
        cpu_sampler<float2> tf_sampler(tf);
        filter_path f(sigma * precision);
        if (midpoint)
            lic_integrate<cpu_sampler<float2>,filter_path,true>(make_float2(ix + 0.5f, iy + 0.5f), tf_sampler, f, tf.w(), tf.h());
        else
            lic_integrate<cpu_sampler<float2>,filter_path,false>(make_float2(ix + 0.5f, iy + 0.5f), tf_sampler, f, tf.w(), tf.h());
        return f.path();
    }

}
