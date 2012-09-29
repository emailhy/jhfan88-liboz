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
#include <oz/stgauss2.h>
#include <oz/st.h>
#include <oz/st_util.h>
#include <oz/cpu_sampler.h>
#include <oz/stintrk.h>
#include <oz/filter_path.h>


namespace oz {

    std::vector<float3> stgauss2_path( int ix, int iy, const cpu_image& st, float sigma, float max_angle,
                                       bool adaptive, bool st_linear, int order, float step_size )
    {
        cpu_sampler<float3> st_sampler(st, st_linear? cudaFilterModeLinear : cudaFilterModePoint);
        float2 p0 = make_float2(ix + 0.5f, iy + 0.5f);
        if (adaptive) {
            float A = st2A(st_sampler(p0.x, p0.y));
            sigma *= 0.25f * (1 + A)*(1 + A);
        }
        float cos_max = cosf(radians(max_angle));
        filter_path f(2 * sigma);
        if (order == 1) st_integrate_euler(p0, st_sampler, f, cos_max, st.w(), st.h(), step_size);
        if (order == 2) st_integrate_rk2(p0, st_sampler, f, cos_max, st.w(), st.h(), step_size);
        if (order == 4) st_integrate_rk4(p0, st_sampler, f, cos_max, st.w(), st.h(), step_size);
        return f.path();
    }

}

