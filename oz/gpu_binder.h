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

#include <oz/gpu_image.h>
#include <oz/type_traits.h>

#ifdef __CUDACC__

namespace oz {

    template<typename T> struct gpu_binder {
        typedef typename type_traits<T>::texture_type TT;
        texture<TT,2>& texref_;

        __host__ gpu_binder( texture<TT,2>& texref, const gpu_image& img,
                             cudaTextureFilterMode filter_mode=cudaFilterModePoint,
                             cudaTextureAddressMode address_mode=cudaAddressModeClamp,
                             bool normalized = false )
                           : texref_(texref)
        {
            OZ_CHECK_FORMAT(img.format(), type_traits<T>::format());
            texref_.filterMode = filter_mode;
            texref_.addressMode[0] = address_mode;
            texref_.addressMode[1] = address_mode;
            texref_.addressMode[2] = address_mode;
            texref_.normalized = (int)normalized;
            OZ_CUDA_SAFE_CALL(cudaBindTexture2D(0, texref_, img.ptr<T>(), img.w(), img.h(), img.pitch()));
        }

        __host__ ~gpu_binder() {
            texref_.filterMode = cudaFilterModePoint;
            texref_.addressMode[0] = cudaAddressModeClamp;
            texref_.addressMode[1] = cudaAddressModeClamp;
            texref_.addressMode[2] = cudaAddressModeClamp;
            texref_.normalized = 0;
            cudaUnbindTexture(texref_);
        }
    };

}

#endif
