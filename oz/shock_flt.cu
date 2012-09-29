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
#include <oz/shock_flt.h>
#include <oz/generate.h>
#include <oz/foreach.h>
#include <oz/gpu_sampler2.h>
#include <oz/gauss.h>
#include <oz/io.h>
#include <sstream>

namespace oz {

    static inline __device__ float minmod( float a, float b ) {
        if (a * b <= 0) return 0;
        return fminf(fabsf(a), fabsf(b));
    }


    struct ShockFlt_OsherRudin : public generator<float> {
        const gpu_sampler<float,0> src_;
        float dt_;

        ShockFlt_OsherRudin( const gpu_image& src, float dt )
            : src_(src), dt_(dt) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float I = src_(ix, iy);

            float I_mx = I - src_(ix - 1, iy);
            float I_px = src_(ix + 1, iy) - I;
            float I_my = I - src_(ix, iy - 1);
            float I_py = src_(ix, iy + 1) - I;

            float I_x = (I_mx + I_px) / 2;
            float I_y = (I_my + I_py) / 2;
            float grad_I2 = I_x * I_x + I_y * I_y;

            float I_xx = I_px - I_mx;
            float I_yy = I_py - I_my;
            float I_xy = (src_(ix + 1, iy + 1) - src_(ix - 1, iy + 1) -
                          src_(ix + 1, iy - 1) + src_(ix - 1, iy - 1)) / 4;

            float I_nn;
            if (grad_I2 > 0) {
                I_nn = (I_xx * I_x*I_x + 2 * I_xy * I_x*I_y + I_yy * I_y*I_y) / grad_I2;
            } else {
                I_nn = I_xx;
            }

            float Dx = minmod(I_mx, I_px);
            float Dy = minmod(I_my, I_py);
            float a_grad_I = sqrtf(Dx * Dx + Dy * Dy);

            return I - dt_ * sign(I_nn) * a_grad_I;
        }
    };


    OZAPI gpu_image shock_flt_or( const gpu_image& src, int N, float dt ) {
        gpu_image tmp = src;
        for (int k = 0; k < N; ++k) {
            tmp = generate(tmp.size(), ShockFlt_OsherRudin(tmp, dt));
        }
        return tmp;
    }


    struct ShockFlt_AlvarezMazorra1 : public generator<float> {
        const gpu_sampler<float,0> src_;
        const gpu_sampler<float,1> src_smooth_;
        float c_;
        float sigma_;
        float dt_;

        struct D {
            float I;
            float I_mx, I_px;
            float I_my, I_py;
            float I_x, I_y;
            float grad_I2;
            float I_xx, I_xy, I_yy;
            float I_nn, I_ee;

            template<typename SRC>
            inline __device__  D( const SRC& src, int ix, int iy ) {
                I = src(ix, iy);
                I_mx = I - src(ix - 1, iy);
                I_px = src(ix + 1, iy) - I;
                I_my = I - src(ix, iy - 1);
                I_py = src(ix, iy + 1) - I;

                I_x = (I_mx + I_px) / 2;
                I_y = (I_my + I_py) / 2;
                grad_I2 = I_x * I_x + I_y * I_y;

                I_xx = I_px - I_mx;
                I_yy = I_py - I_my;
                I_xy = (src(ix + 1, iy + 1) - src(ix - 1, iy + 1) -
                        src(ix + 1, iy - 1) + src(ix - 1, iy - 1)) / 4;

                if (grad_I2 > 0) {
                    I_nn = (I_xx * I_x*I_x + 2 * I_xy * I_x*I_y + I_yy * I_y*I_y) / grad_I2;
                    I_ee = (I_xx * I_y*I_y - 2 * I_xy * I_x*I_y + I_yy * I_x*I_x) / grad_I2;
                } else {
                    I_nn = I_xx;
                    I_ee = I_yy;
                }
            }
        };

        ShockFlt_AlvarezMazorra1( const gpu_image& src, const gpu_image& src_smooth, float c, float sigma, float dt )
            : src_(src), src_smooth_(src_smooth), c_(c), sigma_(sigma), dt_(dt) {}

        inline __device__ float operator()( int ix, int iy ) const {
            D dnn(src_smooth_, ix, iy);
            D dee(src_, ix, iy);

            float Dx = minmod(dee.I_mx, dee.I_px);
            float Dy = minmod(dee.I_my, dee.I_py);
            float a_grad_I = sqrtf(Dx * Dx + Dy * Dy);

            return dee.I + dt_ * (c_ * dee.I_ee - sign(dnn.I_nn) * a_grad_I);
        }
    };


    struct ShockFlt_AlvarezMazorra2_p1 {
        const gpu_sampler<float,0> I_;
        gpu_plm2<float> I_nn_;
        gpu_plm2<float> I_ee_;
        gpu_plm2<float> grad_I_;

        ShockFlt_AlvarezMazorra2_p1( const gpu_image& I, gpu_image& I_nn, gpu_image& I_ee, gpu_image& grad_I )
            : I_(I), I_nn_(I_nn), I_ee_(I_ee), grad_I_(grad_I) {}

        inline __device__ void operator()( int ix, int iy ) {
            float I = I_(ix, iy);
            float I_mx = I - I_(ix - 1, iy);
            float I_px = I_(ix + 1, iy) - I;
            float I_my = I - I_(ix, iy - 1);
            float I_py = I_(ix, iy + 1) - I;

            float I_x = (I_mx + I_px) / 2;
            float I_y = (I_my + I_py) / 2;
            float grad_I2 = I_x * I_x + I_y * I_y;

            float I_xx = I_px - I_mx;
            float I_yy = I_py - I_my;
            float I_xy = ( I_(ix + 1, iy + 1) - I_(ix - 1, iy + 1) -
                           I_(ix + 1, iy - 1) + I_(ix - 1, iy - 1)) / 4;

            float I_nn, I_ee;
            if (grad_I2 > 0) {
                I_nn = (I_xx * I_x*I_x + 2 * I_xy * I_x*I_y + I_yy * I_y*I_y) / grad_I2;
                I_ee = (I_xx * I_y*I_y - 2 * I_xy * I_x*I_y + I_yy * I_x*I_x) / grad_I2;
            } else {
                I_nn = I_xx;
                I_ee = I_yy;
            }

            float Dx = minmod(I_mx, I_px);
            float Dy = minmod(I_my, I_py);
            float a_grad_I = sqrtf(Dx * Dx + Dy * Dy);

            I_nn_.write(ix, iy, I_nn);
            I_ee_.write(ix, iy, I_ee);
            grad_I_.write(ix, iy, a_grad_I);
        }
    };


    struct ShockFlt_AlvarezMazorra2_p2 : public generator<float> {
        const gpu_plm2<float> I_;
        const gpu_plm2<float> I_nn_;
        const gpu_plm2<float> I_ee_;
        const gpu_plm2<float> grad_I_;
        float c_;
        float dt_;

        ShockFlt_AlvarezMazorra2_p2( const gpu_image& I, const gpu_image& I_nn, const gpu_image& I_ee,
                                     const gpu_image& grad_I,  float c, float dt )
            : I_(I), I_nn_(I_nn), I_ee_(I_ee), grad_I_(grad_I), c_(c), dt_(dt) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float I = I_(ix, iy);
            float I_nn = I_nn_(ix, iy);
            float I_ee = I_ee_(ix, iy);
            float grad_I = grad_I_(ix, iy);
            return I + dt_ * (c_ * I_ee - sign(I_nn) * grad_I);
        }
    };


    OZAPI gpu_image shock_flt_am( const gpu_image& src, float c, float sigma, int N, float dt, bool pre_blur ) {
        if (pre_blur) {
            gpu_image tmp = src;
            for (int k = 0; k < N; ++k) {
                gpu_image tmp_smooth = gauss_filter_xy(tmp, sigma, 3);
                tmp = generate(tmp.size(), ShockFlt_AlvarezMazorra1(tmp, tmp_smooth, c, sigma, dt));
            }
            return tmp;
        } else {
            gpu_image I = src;
            gpu_image I_nn(I.size(), FMT_FLOAT);
            gpu_image I_ee(I.size(), FMT_FLOAT);
            gpu_image grad_I(I.size(), FMT_FLOAT);
            for (int k = 0; k < N; ++k) {
                ShockFlt_AlvarezMazorra2_p1 p1(I, I_nn, I_ee, grad_I);
                foreach(I.size(), p1);
                I_nn = gauss_filter_xy(I_nn, sigma, 3);
                I = generate(I.size(), ShockFlt_AlvarezMazorra2_p2(I, I_nn, I_ee, grad_I, c, dt));
            }
            return I;
        }
    }

}
