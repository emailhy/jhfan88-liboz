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
#pragma once

#include <oz/math_util.h>
#include <deque>
#include <vector>

namespace oz {

    class filter_path {
    public:
        filter_path( float radius ) : radius_(radius) {}

        float radius() const {
            return radius_;
        }

        std::vector<float3> path() const {
            return std::vector<float3>(dst_.begin(), dst_.end());
        }

        void operator()(float u, float2 p) {
            if (u < 0)
                dst_.push_front(make_float3(p.x, p.y, u));
            else
                dst_.push_back(make_float3(p.x, p.y, u));
        }

        void operator()(float u, float du, float2 p) {
            if (u < 0)
                dst_.push_front(make_float3(p.x, p.y, u));
            else
                dst_.push_back(make_float3(p.x, p.y, u));
        }

        void error() {
        }

    private:
        std::deque<float3> dst_;
        float radius_;
    };

}
