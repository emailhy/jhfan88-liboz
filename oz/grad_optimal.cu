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
#include <oz/grad_optimal.h>
#include <oz/generate.h>
#include <oz/gpu_sampler2.h>

namespace oz {

    template<typename T> struct GradOptimalSmooth5 : public generator<T> {
        gpu_sampler<T,0> src_;
        bool horizontal_;

        GradOptimalSmooth5( const gpu_image& src, bool horizontal)
            : src_(src), horizontal_(horizontal) {}

        inline __device__ T operator()( int ix, int iy ) const {
            if (horizontal_) {
                return
                    +0.04255f * src_(ix - 2, iy)
                    +0.241f   * src_(ix - 1, iy)
                    +0.4329f  * src_(ix,     iy)
                    +0.241f   * src_(ix + 1, iy)
                    +0.04255f * src_(ix + 2, iy);
            } else {
                return
                    +0.04255f * src_(ix, iy - 2)
                    +0.241f   * src_(ix, iy - 1)
                    +0.4329f  * src_(ix, iy    )
                    +0.241f   * src_(ix, iy + 1)
                    +0.04255f * src_(ix, iy + 2);
            }
        }
    };


    template<typename T> struct GradOptimalFirst5 : public generator<T> {
        gpu_sampler<T,0> src_;
        bool horizontal_;

        GradOptimalFirst5( const gpu_image& src, bool horizontal)
            : src_(src), horizontal_(horizontal) {}

        inline __device__ T operator()( int ix, int iy ) const {
            if (horizontal_) {
                return
                    +0.1f * src_(ix - 2, iy)
                    +0.3f * src_(ix - 1, iy)
                    +0.0f * src_(ix,     iy)
                    -0.3f * src_(ix + 1, iy)
                    -0.1f * src_(ix + 2, iy);
            } else {
                return
                    +0.1f * src_(ix, iy - 2)
                    +0.3f * src_(ix, iy - 1)
                    +0.0f * src_(ix, iy    )
                    -0.3f * src_(ix, iy + 1)
                    -0.1f * src_(ix, iy + 2);
            }
        }
    };


    gpu_image grad_optimal_5x5( const gpu_image& src, bool horizontal ) {
        switch (src.format()) {
            case FMT_FLOAT:
                {
                    gpu_image tmp = generate(src.size(), GradOptimalSmooth5<float>(src, !horizontal));
                    return generate(src.size(), GradOptimalFirst5<float>(tmp, horizontal));
                }
            case FMT_FLOAT3:
                {
                    gpu_image tmp = generate(src.size(), GradOptimalSmooth5<float3>(src, !horizontal));
                    return generate(src.size(), GradOptimalFirst5<float3>(tmp, horizontal));
                }
            default:
                OZ_INVALID_FORMAT();
        }
    }


    template<typename T> struct StOptimal5 : public generator<float3> {
        gpu_sampler<T,0> gx_;
        gpu_sampler<T,1> gy_;

        StOptimal5( const gpu_image& gx, const gpu_image& gy )
            : gx_(gx), gy_(gy) {}

        inline __device__ float3 operator()( int ix, int iy ) const {
            T gx = gx_(ix, iy);
            T gy = gy_(ix, iy);
            return make_float3(dot(gx,gx), dot(gy,gy), dot(gx,gy));
        }
    };


    gpu_image st_optimal_5x5( const gpu_image& src ) {
        gpu_image gx = grad_optimal_5x5(src, true);
        gpu_image gy = grad_optimal_5x5(src, false);
        switch (src.format()) {
            case FMT_FLOAT:  return generate(src.size(), StOptimal5<float >(gx, gy));
            case FMT_FLOAT3: return generate(src.size(), StOptimal5<float3>(gx, gy));
            default:
                OZ_INVALID_FORMAT();
        }
    }
}
