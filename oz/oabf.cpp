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
#include <oz/cpu_image.h>
#include <deque>

namespace oz {

    std::vector<float3> oabf_line( int ix, int iy, const cpu_image& lfm_, float sigma_d_,
                                   float sigma_r_, bool tangential_, float precision_ )
    {
        std::deque<float3> L;

        float4 l = lfm_.at<float4>(ix, iy);
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
        float ds = 1.0f / ((tabs.x > tabs.y)? tabs.x : tabs.y);

        float norm = 1;
        for (float d = ds; d <= halfWidth; d += ds) {
            float2 dt = d * t;

            L.push_back(make_float3(0.5f + ix + dt.x, 0.5f + iy + dt.y, d));
            L.push_front(make_float3(0.5f + ix - dt.x, 0.5f + iy - dt.y, -d));
        }

        return std::vector<float3>(L.begin(), L.end());;
    }


    int oabf_sample_dir( int ix, int iy, const cpu_image& lfm_, bool tangential_ ) {
        float4 l = lfm_.at<float4>(ix, iy);
        float2 t;
        if (tangential_) {
            t = make_float2(l.x, l.y);
        } else {
            t = make_float2(l.y, -l.x);
        }
        float2 tabs = fabs(t);
        return (tabs.x > tabs.y)? 0 : 1;
    }

}
