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
#include <oz/window.h>
#include <oz/generate.h>


namespace oz {

    struct CosineWindow : public oz::generator<float> {
        int width_;
        int height_;
        float radius_;

        CosineWindow( int width, int height, float radius )
            : width_(width), height_(height), radius_(radius) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float x = ix - 0.5f * width_ + 0.5f;
            float y = iy - 0.5f * height_ + 0.5f;
            float r = sqrtf(x*x + y*y) / radius_;
            return (r <= 1)? cosf(0.5f * CUDART_PI_F * r) : 0;
        }
    };

    OZAPI gpu_image cosine_window( int width, int height, float radius ) {
        return generate(width, height, CosineWindow(width, height, radius));
    }

}


