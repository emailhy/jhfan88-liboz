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
#include <oz/curvature.h>
#include <oz/deriv2nd.h>
#include <oz/generate.h>
#include <oz/gpu_sampler5.h>


namespace oz {
    struct imp_curvature : public generator<float> {
        gpu_sampler<float,0> Ix_;
        gpu_sampler<float,1> Iy_;
        gpu_sampler<float,2> Ixx_;
        gpu_sampler<float,3> Ixy_;
        gpu_sampler<float,4> Iyy_;
        float epsilon_;

        imp_curvature( const deriv2nd_t& d, float epsilon )
            : Ix_(d.Ix), Iy_(d.Iy), Ixx_(d.Ixx), Ixy_(d.Ixy) , Iyy_(d.Iyy), epsilon_(epsilon) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float Ix = Ix_(ix, iy);
            float Iy = Iy_(ix, iy);
            float Ixx = Ixx_(ix, iy);
            float Ixy = Ixy_(ix, iy);
            float Iyy = Iyy_(ix, iy);

            const float eps2 = epsilon_ * epsilon_;
            float diff_D = Ixx*(Iy*Iy+eps2) - 2*Ix*Iy*Ixy + Iyy*(Ix*Ix*eps2);
            float diff_N = __powf( Ix*Ix + Iy*Iy + eps2, 1.5f);

            return (diff_N > 0)? diff_D / diff_N : 0;
        }
    };


    gpu_image curvature( const gpu_image& src, float epsilon ) {
        deriv2nd_t d = deriv2nd(src);
        return generate(src.size(), imp_curvature(d, epsilon));
    }
}
