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
#pragma once

#include <oz/gpu_plm2.h>

namespace oz {

    template<typename Result> struct generator {
        typedef Result result_type;
    };

    namespace detail {
        template<typename Generator>
        __global__ void generate_global( gpu_plm2<typename Generator::result_type> dst, const Generator f ) {
            int ix = blockDim.x * blockIdx.x + threadIdx.x;
            int iy = blockDim.y * blockIdx.y + threadIdx.y;
            if (ix >= dst.w || iy >= dst.h) return;
            dst.write(ix, iy, f(ix, iy));
        }
    }

    template<typename Generator>
    void generate_inplace( gpu_image& dst, const Generator& f ) {
        typedef typename Generator::result_type Result;
        OZ_CHECK_FORMAT(dst.format(), type_traits<Result>::format());
        dim3 threads(8, 8);
        dim3 blocks((dst.w()+threads.x-1)/threads.x, (dst.h()+threads.y-1)/threads.y);
        detail::generate_global<Generator><<<blocks, threads>>>(gpu_plm2<Result>(dst), f);
        OZ_CUDA_ERROR_CHECK();
    }

    template<typename Generator>
    gpu_image generate( unsigned w, unsigned h, const Generator& f ) {
        typedef typename Generator::result_type Result;
        gpu_image dst(w, h, type_traits<Result>::format());
        generate_inplace<Generator>(dst, f);
        return dst;
    }

    template<typename Generator>
    gpu_image generate( NppiSize size, const Generator& f ) {
        return generate<Generator>(size.width, size.height, f);
    }

}
