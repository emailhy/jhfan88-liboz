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
#include <oz/prctile.h>
#include <thrust/sort.h>


float oz::prctile( const gpu_image& src, float percent ) {
    if (src.format() != FMT_FLOAT) OZ_INVALID_FORMAT();
    int N = src.N();
    thrust::device_ptr<float> s = thrust::device_malloc<float>(N);
    cudaMemcpy2D(thrust::raw_pointer_cast(s), 4*src.w(), src.ptr(), src.pitch(), 4*src.w(), src.h(), cudaMemcpyDeviceToDevice);
    thrust::sort(s, s+N);

    float result;
    cudaMemcpy(&result, thrust::raw_pointer_cast(s) + (int)(N*percent/100), 4, cudaMemcpyDeviceToHost);
    thrust::device_free(s);

    return result;
}
