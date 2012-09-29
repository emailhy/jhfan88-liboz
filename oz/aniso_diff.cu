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
#include <oz/aniso_diff.h>
#include <oz/generate.h>
#include <oz/gpu_sampler2.h>
#include <oz/gauss.h>

namespace oz {

    static inline __device__ float c(float s, float K, int method) {
        float u = s / K;
        return (method==0)? __expf(-u*u) : 1.0f / (1 + (u*u));
    }


    struct AnisoDiff_PeronaMalik : public generator<float> {
        const gpu_sampler<float,0> src_;
        int method_;
        float K_;
        float dt_;

        AnisoDiff_PeronaMalik( const gpu_image& src, int method, float K, float dt )
            : src_(src), K_(K), method_(method), dt_(dt) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float I = src_(ix, iy);
            float In = src_(ix, iy - 1) - I;
            float Is = src_(ix, iy + 1) - I;
            float Ie = src_(ix + 1, iy) - I;
            float Iw = src_(ix - 1, iy) - I;

            float Cn = c(In, K_, method_);
            float Cs = c(Is, K_, method_);
            float Ce = c(Ie, K_, method_);
            float Cw = c(Iw, K_, method_);

            return I + dt_ * (Cn * In + Cs * Is + Ce * Ie + Cw * Iw);
        }
    };


    struct AnisoDiff_CatteEtAl : public generator<float> {
        const gpu_sampler<float,0> src_;
        const gpu_sampler<float,1> src_smooth_;
        int method_;
        float K_;
        float dt_;

        AnisoDiff_CatteEtAl( const gpu_image& src, const gpu_image& src_smooth, int method, float K, float dt )
            : src_(src), src_smooth_(src_smooth), K_(K), method_(method), dt_(dt) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float I = src_smooth_(ix, iy);
            float In = src_smooth_(ix, iy - 1) - I;
            float Is = src_smooth_(ix, iy + 1) - I;
            float Ie = src_smooth_(ix + 1, iy) - I;
            float Iw = src_smooth_(ix - 1, iy) - I;

            float Cn = c(In, K_, method_);
            float Cs = c(Is, K_, method_);
            float Ce = c(Ie, K_, method_);
            float Cw = c(Iw, K_, method_);

            I = src_(ix, iy);
            In = src_(ix, iy - 1) - I;
            Is = src_(ix, iy + 1) - I;
            Ie = src_(ix + 1, iy) - I;
            Iw = src_(ix - 1, iy) - I;

            return I + dt_ * (Cn * In + Cs * Is + Ce * Ie + Cw * Iw);
        }
    };


    OZAPI gpu_image aniso_diff( const gpu_image& src, int method, float K, float sigma, int N, float dt ) {
        if ((method < 0) || (method > 1)) OZ_X() << "Invalid method!";
        if (sigma > 0) {
            gpu_image tmp = src;
            for (int k = 0; k < N; ++k) {
                gpu_image tmp_smooth = gauss_filter_xy(tmp, sigma, 3.0f);
                tmp = generate(tmp.size(), AnisoDiff_CatteEtAl(tmp, tmp_smooth, method, K, dt));
            }
            return tmp;
        } else {
            gpu_image tmp = src;
            for (int k = 0; k < N; ++k) {
                tmp = generate(tmp.size(), AnisoDiff_PeronaMalik(tmp, method, K, dt));
            }
            return tmp;
        }
    }
}
