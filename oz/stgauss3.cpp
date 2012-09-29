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
#include <oz/stgauss3.h>
#include <oz/filter_path.h>
#include <oz/cpu_sampler.h>
#include <oz/stintrk2.h>

namespace oz {

    std::vector<float3> stgauss3_path_( int ix, int iy, const cpu_image& st, float sigma,
                                       bool st_linear, bool adaptive, bool ustep,
                                       int order, float step_size )
    {
        cpu_sampler<float3> st_sampler(st, st_linear? cudaFilterModeLinear : cudaFilterModePoint);
        float2 p0 = make_float2(ix + 0.5f, iy + 0.5f);
        filter_path f(2 * sigma);
        if (!ustep) {
            if (order == 1) {
                if (adaptive)
                    st3_int<cpu_sampler<float3>,filter_path,1,true>(p0, st_sampler, f, st.w(), st.h(), step_size);
                else
                    st3_int<cpu_sampler<float3>,filter_path,1,false>(p0, st_sampler, f, st.w(), st.h(), step_size);
            } else {
                if (adaptive)
                    st3_int<cpu_sampler<float3>,filter_path,2,true>(p0, st_sampler, f, st.w(), st.h(), step_size);
                else
                    st3_int<cpu_sampler<float3>,filter_path,2,false>(p0, st_sampler, f, st.w(), st.h(), step_size);
            }
        } else {
            if (order == 1) {
                st3_int_ustep<cpu_sampler<float3>,filter_path,1>(p0, st_sampler, f, st.w(), st.h(), step_size);
            } else {
                st3_int_ustep<cpu_sampler<float3>,filter_path,2>(p0, st_sampler, f, st.w(), st.h(), step_size);
            }
        }
        return f.path();
    }

}
