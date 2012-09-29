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

#include <oz/cpu_image.h>
#include <oz/type_traits.h>

namespace oz {

    template<typename T>
    struct cpu_sampler {
        const cpu_image& img_;
        const cudaTextureFilterMode filter_mode_;

        cpu_sampler(const cpu_image& img, cudaTextureFilterMode filter_mode=cudaFilterModePoint)
            : img_(img), filter_mode_(filter_mode)
        {
            if (img.format() != type_traits<T>::format()) OZ_INVALID_FORMAT();
        }

        unsigned w() const {
            return img_.w();
        }

        unsigned h() const {
            return img_.h();
        }

        T operator()(float x, float y) const {
            return (filter_mode_ == cudaFilterModePoint)?
                img_.sample_nearest<T>(x, y) : img_.sample_linear<T>(x, y);
        }

        T operator()( float2 v ) const {
            return operator()(v.x, v.y);
        }
    };

}
