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
#include <oz/convpyr.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>
#include <oz/pad.h>
#include <vector>
#include <oz/blit.h>

namespace oz {

    template<typename T, int pass> struct ConvPyrDown : public generator<T> {
        gpu_sampler<T,0> src_;

        ConvPyrDown( const gpu_image& src) : src_(src) {}

        __device__ T operator()( int ix, int iy ) const {
            const float W[6] = { 0.1507, 0.6836, 1.0334, 0.0270, 0.0312, 0.7753 };
            const float H1[5] = { W[0], W[1], W[2], W[1], W[0] };
            if (pass == 0) {
                int ux = 2 * ix;
                return H1[0] * src_(ux - 2, iy) +
                       H1[1] * src_(ux - 1, iy) +
                       H1[2] * src_(ux,     iy) +
                       H1[3] * src_(ux + 1, iy) +
                       H1[4] * src_(ux + 2, iy);
            } else {
                int uy = 2 * iy;
                return H1[0] * src_(ix, uy - 2) +
                       H1[1] * src_(ix, uy - 1) +
                       H1[2] * src_(ix, uy    ) +
                       H1[3] * src_(ix, uy + 1) +
                       H1[4] * src_(ix, uy + 2);
            }
        }
    };

    template<typename T> gpu_image convpyr_down( const gpu_image& src ) {
        int w = (src.w() + 1) / 2;
        int h = (src.h() + 1) / 2;
        gpu_image tmp = generate(w, src.h(), ConvPyrDown<T,0>(src));
        return generate(w, h, ConvPyrDown<T,1>(tmp));
    }


    template<typename T, int pass> struct ConvPyrUp : public generator<T> {
        gpu_sampler<T,0> src_;

        ConvPyrUp( const gpu_image& src) : src_(src) {}

        __device__ T operator()( int ix, int iy ) const {
            const float W[6] = { 0.1507, 0.6836, 1.0334, 0.0270, 0.0312, 0.7753 };
            const float H2[5] = { W[0]*sqrtf(W[3]), W[1]*sqrtf(W[3]), W[2]*sqrtf(W[3]), W[1]*sqrtf(W[3]), W[0]*sqrtf(W[3]) };
            if (pass == 0) {
                int ux = ix / 2;
                T c;
                if (ix & 1) {
                    c = H2[1] * src_(ux,     iy) +
                        H2[3] * src_(ux + 1, iy);
                } else {
                    c = H2[0] * src_(ux - 1, iy) +
                        H2[2] * src_(ux,     iy) +
                        H2[4] * src_(ux + 1, iy);
                }
                return c;
            } else {
                int uy = iy / 2;
                T c;
                if (iy & 1) {
                    c = H2[1] * src_(ix, uy    ) +
                        H2[3] * src_(ix, uy + 1);
                } else {
                    c = H2[0] * src_(ix, uy - 1) +
                        H2[2] * src_(ix, uy    ) +
                        H2[4] * src_(ix, uy + 1);
                }
                return c;
            }
        }
    };

    template<typename T> gpu_image convpyr_up( const gpu_image& src, unsigned w, unsigned h ) {
        if (!w) w = 2 * src.w();
        if (!h) h = 2 * src.h();
        gpu_image tmp = generate(w, src.h(), ConvPyrUp<T,0>(src));
        return generate(w, h, ConvPyrUp<T,1>(tmp));
    }


    template<typename T> struct ConvPyrG : public generator<T> {
        gpu_sampler<T,0> src_;

        ConvPyrG( const gpu_image& src) : src_(src) {}

        __device__ T operator()( int ix, int iy ) const {
            const float W[6] = { 0.1507, 0.6836, 1.0334, 0.0270, 0.0312, 0.7753 };
            const float G[3] = { W[4], W[5], W[4] };
            T c = make_zero<T>();
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    c += G[i+1] * G[j+1] * src_(ix+i, iy+j);

                }
            }
            return c;
        }
    };

    template<typename T> gpu_image convpyr_g( const gpu_image& src  ) {
        return generate(src.w(), src.h(), ConvPyrG<T>(src));
    }


    struct ConvPyrSetup : public generator<float4> {
        gpu_sampler<float4,0> src_;

        ConvPyrSetup( const gpu_image& src ) : src_(src) {}

        __device__ float4 operator()( int ix, int iy ) const {
            float4 c = src_(ix, iy);
            if (c.w >= 1) {
                float w = src_(ix+1, iy  ).w +
                          src_(ix,   iy+1).w +
                          src_(ix-1, iy  ).w +
                          src_(ix,   iy-1).w;
                if (w < 4) {
                    return make_float4(make_float3(c), 1);
                }
            }
            return make_float4(0);
        }
    };


    struct ConvPyrHomog : public generator<float3> {
        gpu_plm2<float4> org_;
        gpu_plm2<float4> src_;

        ConvPyrHomog( const gpu_image& org, const gpu_image& src ) : org_(org), src_(src) {}

        __device__ float3 operator()( int ix, int iy ) const {
            float4 c = org_(ix, iy);
            if (c.w >= 1) {
                return make_float3(c);
            }
            c = src_(ix, iy);
            return (c.w > 0)? make_float3(c) / c.w : make_float3(0);
        }
    };


    gpu_image convpyr_boundary( const gpu_image& src ) {
        gpu_image a = generate(src.w(), src.h(), ConvPyrSetup(src));

        int maxLevel = (int)ceil(log2((double)max(a.w(), a.h())));
        std::vector<gpu_image> pyr;
        pyr.resize(maxLevel);

        // Forward transform (analysis)
        pyr[0] = padzero(a, 5,5);
        for (int i = 1; i < maxLevel; ++i) {
            gpu_image down = convpyr_down<float4>(pyr[i-1]);
            pyr[i] = padzero(down, 5,5);
        }

        // Backward transform (synthesis)
        std::vector<gpu_image> fpyr;
        fpyr.resize(maxLevel);

        fpyr[maxLevel-1] = convpyr_g<float4>(pyr[maxLevel-1]);
        for (int i = maxLevel-2; i >= 0; --i) {

            gpu_image rd = fpyr[i + 1];
            rd = unpad(rd, 5, 5);

            gpu_image G = convpyr_g<float4>(pyr[i]);
            gpu_image U = convpyr_up<float4>(rd, pyr[i].w(), pyr[i].h());
            fpyr[i] = U + G;
        }

        gpu_image r = unpad(fpyr[0], 5,5);
        return generate(src.w(), src.h(), ConvPyrHomog(src,r));
    }

}
