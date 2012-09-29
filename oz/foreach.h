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

namespace oz {

    namespace detail {
        template<typename Functor>
        __global__ void foreach_global( unsigned w, unsigned h, Functor f ) {
            const int ix = blockDim.x * blockIdx.x + threadIdx.x;
            const int iy = blockDim.y * blockIdx.y + threadIdx.y;
            if(ix >= w || iy >= h) return;
            f(ix, iy);
        }
    }

    template<typename Functor>
    void foreach( unsigned w, unsigned h, Functor& f ) {
        dim3 threads(8, 8);
        dim3 blocks((w+threads.x-1)/threads.x, (h+threads.y-1)/threads.y);
        detail::foreach_global<Functor><<<blocks, threads>>>(w, h, f);
        OZ_CUDA_ERROR_CHECK();
    }

    template<typename Functor>
    void foreach( NppiSize size, Functor& f ) {
        foreach<Functor>(size.width, size.height, f);
    }

}
