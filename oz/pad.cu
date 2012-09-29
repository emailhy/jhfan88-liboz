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
#include <oz/pad.h>
#include <oz/generate.h>

namespace oz {

    template<typename T> struct PadZero : public generator<T> {
        gpu_plm2<T> src_;
        int dx_;
        int dy_;

        PadZero( const gpu_image& src, int dx, int dy)
            : src_(src), dx_(dx), dy_(dy) {}

        __device__ T operator()( int ix, int iy ) const {
            if ((ix < dx_) || (iy < dy_) || (ix >= src_.w + dx_) || (iy >= src_.h + dy_)) {
                return make_zero<T>();
            }
            return src_(ix - dx_, iy - dy_);
        }
    };

    gpu_image padzero( const gpu_image& src, int dx, int dy ) {
        switch (src.format()) {
            case FMT_FLOAT:  return generate(src.w() + 2*dx, src.h() + 2* dy, PadZero<float >(src, dx, dy));
            case FMT_FLOAT2: return generate(src.w() + 2*dx, src.h() + 2* dy, PadZero<float2>(src, dx, dy));
            case FMT_FLOAT3: return generate(src.w() + 2*dx, src.h() + 2* dy, PadZero<float3>(src, dx, dy));
            case FMT_FLOAT4: return generate(src.w() + 2*dx, src.h() + 2* dy, PadZero<float4>(src, dx, dy));
            default:
                OZ_INVALID_FORMAT();
        }
    }


    template<typename T> struct Unpad : public generator<T> {
        gpu_plm2<T> src_;
        int dx_;
        int dy_;

        Unpad( const gpu_image& src, int dx, int dy)
            : src_(src), dx_(dx), dy_(dy) {}

        __device__ T operator()( int ix, int iy ) const {
            return src_(ix + dx_, iy + dy_);
        }
    };

    gpu_image unpad( const gpu_image& src, int dx, int dy ) {
        switch (src.format()) {
            case FMT_FLOAT:  return generate(src.w() - 2*dx, src.h() - 2* dy, Unpad<float >(src, dx, dy));
            case FMT_FLOAT2: return generate(src.w() - 2*dx, src.h() - 2* dy, Unpad<float2>(src, dx, dy));
            case FMT_FLOAT3: return generate(src.w() - 2*dx, src.h() - 2* dy, Unpad<float3>(src, dx, dy));
            case FMT_FLOAT4: return generate(src.w() - 2*dx, src.h() - 2* dy, Unpad<float4>(src, dx, dy));
            default:
                OZ_INVALID_FORMAT();
        }
    }

}
