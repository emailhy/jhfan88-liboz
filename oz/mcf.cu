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
#include <oz/mcf.h>
#include <oz/deriv2nd.h>
#include <oz/generate.h>
#include <oz/gpu_sampler6.h>


namespace oz {
    struct imp_mcf : public generator<float> {
        gpu_sampler<float,0> src_;
        gpu_sampler<float,1> Ix_;
        gpu_sampler<float,2> Iy_;
        gpu_sampler<float,3> Ixx_;
        gpu_sampler<float,4> Ixy_;
        gpu_sampler<float,5> Iyy_;
        float step_;
        float epsilon_;

        imp_mcf( const gpu_image& src, const deriv2nd_t& d, float step, float epsilon )
            : src_(src), Ix_(d.Ix), Iy_(d.Iy), Ixx_(d.Ixx), Ixy_(d.Ixy) , Iyy_(d.Iyy),
              step_(step), epsilon_(epsilon) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float I = src_(ix, iy);
            float Ix = Ix_(ix, iy);
            float Iy = Iy_(ix, iy);
            float Ixx = Ixx_(ix, iy);
            float Ixy = Ixy_(ix, iy);
            float Iyy = Iyy_(ix, iy);

            float eps2 = epsilon_ * epsilon_;
            float diff_D = Ixx*(Iy*Iy+eps2) - 2*Ix*Iy*Ixy + Iyy*(Ix*Ix+eps2);
            float diff_N = (Ix*Ix + Iy*Iy + eps2);

            if (diff_N > 0) {
                //double sq = sqrtf(diff_N);
                //double kappa =  diff_D / (sq * sq * sq);
                //I += step_ * sq  * kappa;
                I += step_ * diff_D / diff_N;
            }
            return I;
        }
    };


    gpu_image mcf( const gpu_image& src, float step, float epsilon ) {
        deriv2nd_t d = deriv2nd(src);
        return generate(src.size(), imp_mcf(src, d, step, epsilon));
    }
}
