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
#include <oz/deriv2nd.h>
#include <oz/foreach.h>
#include <oz/generate.h>
#include <oz/gpu_sampler2.h>
#include <oz/gpu_plm2.h>

namespace oz {

    template<typename T> struct imp_deriv2nd {
        const gpu_sampler<T,0> I_;
        gpu_plm2<T> I_x_;
        gpu_plm2<T> I_y_;
        gpu_plm2<T> I_xx_;
        gpu_plm2<T> I_xy_;
        gpu_plm2<T> I_yy_;

        imp_deriv2nd( const gpu_image& I,
                      gpu_image& I_x, const gpu_image& I_y,
                      gpu_image& I_xx, gpu_image& I_xy, gpu_image& I_yy )
            : I_(I), I_x_(I_x), I_y_(I_y), I_xx_(I_xx), I_xy_(I_xy), I_yy_(I_yy) {}

        inline __device__ void operator()( int ix, int iy ) {
            T I = I_(ix, iy);

            T I_mx = I - I_(ix - 1, iy);
            T I_px = I_(ix + 1, iy) - I;
            T I_my = I - I_(ix, iy - 1);
            T I_py = I_(ix, iy + 1) - I;

            I_x_.write(ix, iy, (I_mx + I_px) / 2);
            I_y_.write(ix, iy, (I_my + I_py) / 2);

            I_xx_.write(ix, iy, I_px - I_mx);
            I_yy_.write(ix, iy, I_py - I_my);
            I_xy_.write(ix, iy, (I_(ix + 1, iy + 1) - I_(ix - 1, iy + 1) -
                                 I_(ix + 1, iy - 1) + I_(ix - 1, iy - 1)) / 4);
        }
    };


    template<typename T>
    static deriv2nd_t deriv2ndT( const gpu_image& src ) {
        deriv2nd_t d(src.w(), src.h(), src.format());
        imp_deriv2nd<T> f(src, d.Ix, d.Iy, d.Ixx, d.Ixy, d.Iyy);
        foreach< imp_deriv2nd<T> >(src.size(), f);
        return d;
    }


    template<typename T> struct imp_deriv2nd_sign : public generator<float> {
        const gpu_plm2<float2> dir_;
        const gpu_plm2<T> Ixx_;
        const gpu_plm2<T> Ixy_;
        const gpu_plm2<T> Iyy_;

        imp_deriv2nd_sign( const gpu_image& dir, const gpu_image& Ixx,
                           const gpu_image& Ixy, const gpu_image& Iyy )
            : dir_(dir), Ixx_(Ixx), Ixy_(Ixy), Iyy_(Iyy) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float2 d = dir_(ix, iy);
            T vxx = Ixx_(ix, iy);
            T vxy = Ixy_(ix, iy);
            T vyy = Iyy_(ix, iy);

            float c = d.x;
            float s = d.y;
            T sign = c*c * vxx + 2*c*s * vxy + s*s *vyy;

            return -sum(sign);
        }
    };
}


oz::deriv2nd_t oz::deriv2nd( const gpu_image& src ) {
    switch (src.format()) {
        case FMT_FLOAT:  return deriv2ndT<float >(src);
        case FMT_FLOAT3: return deriv2ndT<float3>(src);
        default:
            OZ_INVALID_FORMAT();
    }
}


oz::gpu_image oz::deriv2nd_sign( const gpu_image& dir, const gpu_image& Ixx,
                                 const gpu_image& Ixy, const gpu_image& Iyy )
{
    switch (Ixx.format()) {
        case FMT_FLOAT:  return generate(Ixx.size(), imp_deriv2nd_sign<float >(dir, Ixx, Ixy, Iyy));
        case FMT_FLOAT3: return generate(Ixx.size(), imp_deriv2nd_sign<float3>(dir, Ixx, Ixy, Iyy));
        default:
            OZ_INVALID_FORMAT();
    }
}
