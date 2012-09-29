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
#include <oz/median.h>
#include <oz/generate.h>
#include <oz/gpu_sampler1.h>


namespace {
    template<typename T> struct Median3x3;


    template<> struct Median3x3<float> : public oz::generator<float> {
        gpu_sampler<float,0> src_;
        Median3x3( const oz::gpu_image& src ) : src_(src) {}

        inline __device__ float operator()( int ix, int iy ) const {
            float tmp[9];
            for (int i =0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    tmp[3*i+j] = src_( ix + j - 1, iy + i - 1);
                }
            }

            const int N = 9;
            for (int i = 0; i < N; i++) {
                for (int j = N-1; j > i; j--) {
                    if (tmp[j-1] > tmp[j]) {
                        float t = tmp[j-1];
                        tmp[j-1] = tmp[j];
                        tmp[j] = t;
                    }
                }
            }

            return tmp[4];
        }
    };


    template<> struct Median3x3<float2> : public oz::generator<float2> {
        gpu_sampler<float2,0> src_;
        Median3x3( const oz::gpu_image& src ) : src_(src) {}

        inline __device__ float2 operator()( int ix, int iy ) const {
            float tmp[2][9];
            for (int i =0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    float2 v = src_(ix + j - 1, iy + i - 1);
                    tmp[0][3*i+j] = v.x;
                    tmp[1][3*i+j] = v.y;
                }
            }

            const int N = 9;
            for (int k = 0; k < 2; ++k) {
                for (int i = 0; i < N; i++) {
                    for (int j = N-1; j > i; j--) {
                        if (tmp[k][j-1] > tmp[k][j]) {
                            float t = tmp[k][j-1];
                            tmp[k][j-1] = tmp[k][j];
                            tmp[k][j] = t;
                        }
                    }
                }
            }

            return make_float2(tmp[0][4], tmp[1][4]);
        }
    };
}

oz::gpu_image oz::median_3x3( const gpu_image& src ) {
    switch (src.format()) {
        case FMT_FLOAT:   return generate(src.size(), Median3x3<float >(src));
        case FMT_FLOAT2:  return generate(src.size(), Median3x3<float2>(src));
        default:
            OZ_INVALID_FORMAT();
    }
}
